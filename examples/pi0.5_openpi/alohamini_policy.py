import dataclasses
from typing import ClassVar
import einops
import numpy as np

from openpi import transforms


def _api16_to_internal17(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 16:
        raise ValueError(f"Expected 16D input for api16_to_internal17, got shape {x.shape}")

    x18 = np.insert(x, 5, 0.0, axis=-1)  # 17D: theta at 15, lift at 16 (after this step)
    x18 = np.insert(x18, 12, 0.0, axis=-1)  # 18D: theta at 16, lift at 17
    x17 = np.delete(x18, 16, axis=-1)
    x17[..., 5] = 0.0
    x17[..., 12] = 0.0
    return x17


def _internal17_to_api16(actions17: np.ndarray) -> np.ndarray:
    if actions17.shape[-1] != 17:
        raise ValueError(f"Expected 17D internal actions, got shape {actions17.shape}")

    actions17 = np.asarray(actions17)
    actions17[..., 5] = 0.0
    actions17[..., 12] = 0.0

    h = actions17.shape[-2] if actions17.ndim >= 2 else 1
    a2 = actions17.reshape((h, 17))
    theta0 = np.zeros((h, 1), dtype=a2.dtype)
    api16 = np.concatenate(
        [
            a2[:, 0:5],   
            a2[:, 6:7],   
            a2[:, 7:12],  
            a2[:, 13:14], 
            a2[:, 14:16], 
            theta0,       
            a2[:, 16:17], 
        ],
        axis=-1,
    )
    return api16


def _api16_to_internal18(x: np.ndarray) -> np.ndarray:
    if x.shape[-1] != 16:
        raise ValueError(f"Expected 16D input for api16_to_internal18, got shape {x.shape}")
    x18 = np.insert(x, 5, 0.0, axis=-1)   
    x18 = np.insert(x18, 12, 0.0, axis=-1)  
    x18[..., 5] = 0.0
    x18[..., 12] = 0.0
    return x18


def _internal18_to_api16(actions18: np.ndarray) -> np.ndarray:
    if actions18.shape[-1] != 18:
        raise ValueError(f"Expected 18D internal actions, got shape {actions18.shape}")

    actions18 = np.asarray(actions18)
    actions18[..., 5] = 0.0
    actions18[..., 12] = 0.0

    h = actions18.shape[-2] if actions18.ndim >= 2 else 1
    a2 = actions18.reshape((h, 18))
    api16 = np.concatenate(
        [
            a2[:, 0:5],    
            a2[:, 6:7],    
            a2[:, 7:12],   
            a2[:, 13:14],  
            a2[:, 14:16],  
            a2[:, 16:17],  
            a2[:, 17:18],  
        ],
        axis=-1,
    )
    return api16


@dataclasses.dataclass(frozen=True)
class AddVirtualJoint6(transforms.DataTransformFn):    
    def __call__(self, data: dict) -> dict:
        result = dict(data)
        
        if "state" in result:
            state = np.asarray(result["state"], dtype=np.float32)
            if state.shape[-1] == 16:
                result["state"] = _api16_to_internal17(state)
            elif state.shape[-1] == 18:
                state_17 = np.delete(state, 16, axis=-1)
                state_17[..., 5] = 0.0
                state_17[..., 12] = 0.0
                result["state"] = state_17
            elif state.shape[-1] == 17:
                state[..., 5] = 0.0
                state[..., 12] = 0.0
                result["state"] = state
        
        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape[-1] == 16:
                result["actions"] = _api16_to_internal17(actions)
            elif actions.shape[-1] == 18:
                actions_17 = np.delete(actions, 16, axis=-1)
                actions_17[..., 5] = 0.0
                actions_17[..., 12] = 0.0
                result["actions"] = actions_17
            elif actions.shape[-1] == 17:
                actions[..., 5] = 0.0
                actions[..., 12] = 0.0
                result["actions"] = actions
        
        return result


@dataclasses.dataclass(frozen=True)
class AddVirtualJoint6Legacy(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        result = dict(data)

        if "state" in result:
            state = np.asarray(result["state"], dtype=np.float32)
            if state.shape[-1] == 16:
                result["state"] = _api16_to_internal18(state)
            elif state.shape[-1] == 18:
                state[..., 5] = 0.0
                state[..., 12] = 0.0
                result["state"] = state

        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape[-1] == 16:
                result["actions"] = _api16_to_internal18(actions)
            elif actions.shape[-1] == 18:
                actions[..., 5] = 0.0
                actions[..., 12] = 0.0
                result["actions"] = actions

        return result


@dataclasses.dataclass(frozen=True)
class RemoveVirtualJoint6(transforms.DataTransformFn):

    def __call__(self, data: dict) -> dict:
        result = dict(data)
        
        # Process actions: [H, 17] -> [H, 16]
        if "actions" in result:
            actions = np.asarray(result["actions"], dtype=np.float32)
            if actions.shape[-1] == 17:
                result["actions"] = _internal17_to_api16(actions)
        
        return result


@dataclasses.dataclass(frozen=True)
class AlohaMiniInputs(transforms.DataTransformFn):
    bgr_to_rgb: bool = False

    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = (
        "cam_high", "cam_left_wrist", "cam_right_wrist", "cam_low")

    def __call__(self, data: dict) -> dict:
        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(
                f"Unexpected image keys: {tuple(in_images)}. Expected subset of {self.EXPECTED_CAMERAS}.")

        def convert_image(img):
            img = np.asarray(img)
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            hwc = einops.rearrange(img, "c h w -> h w c")
            if self.bgr_to_rgb and hwc.ndim == 3 and hwc.shape[-1] == 3:
                hwc = hwc[..., ::-1].copy()
            return hwc

        images_dict = {name: convert_image(img)
                       for name, img in in_images.items()}

        base_image = images_dict["cam_high"]
        images = {"base_0_rgb": base_image}
        image_masks = {"base_0_rgb": np.True_}

        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in images_dict:
                images[dest] = images_dict[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        out = {
            "image": images,
            "image_mask": image_masks,
            "state": np.asarray(data["state"], dtype=np.float32),
        }

        if "actions" in data:
            out["actions"] = np.asarray(data["actions"], dtype=np.float32)

        if "prompt" in data:
            out["prompt"] = data["prompt"]

        return out


@dataclasses.dataclass(frozen=True)
class AlohaMiniOutputs(transforms.DataTransformFn):
    internal_dim: int | None = 17

    def __call__(self, data: dict) -> dict:
        actions = np.asarray(data["actions"], dtype=np.float32)
        internal_dim = int(self.internal_dim or 17)
        if internal_dim not in (17, 18):
            raise ValueError(f"AlohaMiniOutputs: internal_dim must be 17 or 18, got {internal_dim}")
        if actions.shape[-1] < internal_dim:
            raise ValueError(
                f"AlohaMiniOutputs: actions dimension ({actions.shape[-1]}) is smaller than internal_dim={internal_dim}. "
                f"Expected actions to have at least {internal_dim} dimensions after AbsoluteActions."
            )
        if internal_dim == 18:
            internal = actions[:, :18].copy()
            internal[:, 5] = 0.0
            internal[:, 12] = 0.0
            api16 = _internal18_to_api16(internal)
        else:
            internal = actions[:, :17].copy()
            internal[:, 5] = 0.0
            internal[:, 12] = 0.0
            api16 = _internal17_to_api16(internal)

        api16 = np.asarray(api16, dtype=np.float32)
        api16[:, 0:5] = np.clip(api16[:, 0:5], -100.0, 100.0)   
        api16[:, 6:11] = np.clip(api16[:, 6:11], -100.0, 100.0) 
        api16[:, 5] = np.clip(api16[:, 5], 0.0, 100.0)          
        api16[:, 11] = np.clip(api16[:, 11], 0.0, 100.0)        
        api16[:, 12:14] = np.clip(api16[:, 12:14], -0.15, 0.15) 
        api16[:, 15] = np.clip(api16[:, 15], 0.0, 0.45)         
        
        import logging
        logger = logging.getLogger(__name__)
        if not hasattr(self, "_logged_once"):
            logger.debug(
                f"AlohaMiniOutputs INPUT: actions shape={actions.shape}, "
                f"internal range=[{internal.min():.3f}, {internal.max():.3f}], "
                f"internal mean_abs={np.abs(internal).mean():.3f}, "
                f"internal_dim={internal_dim}"
            )
            object.__setattr__(self, "_logged_once", True)
        
        if not hasattr(self, "_logged_output_once"):
            logger.debug(
                f"AlohaMiniOutputs OUTPUT: api16 shape={api16.shape}, "
                f"range=[{api16.min():.3f}, {api16.max():.3f}], "
                f"mean_abs={np.abs(api16).mean():.3f}"
            )
            if np.any(np.abs(api16) > 1000.0):
                logger.warning(
                    f"AlohaMiniOutputs: CRITICAL - api16 actions extremely large! "
                    f"max_abs={np.abs(api16).max():.3f}"
                )
            elif np.any(np.abs(api16) > 100.0):
                logger.warning(
                    f"AlohaMiniOutputs: WARNING - api16 actions very large! "
                    f"max_abs={np.abs(api16).max():.3f}"
                )
            object.__setattr__(self, "_logged_output_once", True)
        
        return {"actions": api16}

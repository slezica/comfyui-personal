import time
import itertools
import collections
import re
import torch
import torch.nn.functional as F
import math
import weakref
import types
from pprint import pprint

import folder_paths
import comfy
from comfy.model_management import get_torch_device
from comfy_execution.graph_utils import GraphBuilder
from comfy_extras.nodes_canny import Canny
from comfy_extras.nodes_controlnet import SetUnionControlNetType
from comfy_extras.nodes_pag import PerturbedAttentionGuidance
from comfy_extras.nodes_mask import ImageCompositeMasked, InvertMask, GrowMask
from comfy_extras.nodes_upscale_model import UpscaleModelLoader, ImageUpscaleWithModel
from comfy_extras.nodes_differential_diffusion import DifferentialDiffusion
from comfy_extras.nodes_custom_sampler import (
    SamplerCustom,
    SamplerCustomAdvanced,
    RandomNoise,
    CFGGuider,
    BasicScheduler,
    KSamplerSelect,
)
from comfy_extras.nodes_sd3 import ControlNetApplySD3
from nodes import (
    EmptyImage,
    LoadImage,
    SaveImage,
    KSampler,
    VAEDecode,
    VAEEncode,
    PreviewImage,
    ImageScaleBy,
    CheckpointLoaderSimple,
    LoraLoader,
    LoraLoaderModelOnly,
    CLIPVisionLoader,
    CLIPTextEncode,
    CLIPVisionEncode,
    InpaintModelConditioning,
    EmptyLatentImage,
    ImageScale,
    ImagePadForOutpaint,
    ControlNetLoader,
    ControlNetApplyAdvanced,
    StyleModelApply,
    StyleModelLoader,
    ConditioningCombine,
    ConditioningSetMask,
    MAX_RESOLUTION
)
from custom_nodes.ComfyUI_InstantID.InstantID import (
    InstantIDModelLoader,
    ApplyInstantIDAdvanced,
    InstantIDFaceAnalysis
)

from custom_nodes.comfyui_tcd import TCDModelSamplingDiscrete
from custom_nodes.ComfyUI_IPAdapter_plus.IPAdapterPlus import IPAdapterModelLoader, IPAdapterAdvanced
from custom_nodes.comfyui_impact_pack import GaussianBlurMask
from custom_nodes.ComfyUI_InfiniteYou import (
    IDEmbeddingModelLoader,
    ExtractFacePoseImage,
    ExtractIDEmbedding,
    InfuseNetLoader,
    InfuseNetApply
)


class Ref:
    def __init__(self, value):
        self.value = value

wcache = weakref.WeakValueDictionary()


# --------------------------------------------------------------------------------------------------

class Personal:
    CATEGORY = "personal"
    FUNCTION = "execute"


# --------------------------------------------------------------------------------------------------

class UseCheckpoint(Personal):

    @classmethod
    def INPUT_TYPES(cls):
        inputs = CheckpointLoaderSimple.INPUT_TYPES()
        return {
            'required': {'f': inputs['required']['ckpt_name']},
        }

    RETURN_TYPES = ('MODEL', 'CLIP', 'VAE')

    def execute(self, f):
        g = GraphBuilder()
        g_loader = g.node('CheckpointLoaderSimple', ckpt_name=f)

        return {
            'result': (g_loader.out(0), g_loader.out(1), g_loader.out(2)),
            'expand': g.finalize()
        }


# --------------------------------------------------------------------------------------------------

class UseLora(Personal):

    @classmethod
    def INPUT_TYPES(s):
        ll = LoraLoader.INPUT_TYPES()
        return {
            "required": {
                'model': ll['required']['model'],
                'clip': ll['required']['clip'],
                'f': ll['required']['lora_name'],
                "strength": ("FLOAT", {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05}),
            }
        }

    RETURN_TYPES = ('MODEL', 'CLIP')

    def execute(self, model, clip, f, strength):
        g = GraphBuilder()

        g_lora = g.node(
            'LoraLoader',
            model          = model,
            clip           = clip,
            lora_name      = f,
            strength_model = strength,
            strength_clip  = 1.0
        )

        return {
            'result': (g_lora.out(0), g_lora.out(1)),
            'expand': g.finalize()
        }


# --------------------------------------------------------------------------------------------------

class UseControlNet(Personal):

    @classmethod
    def INPUT_TYPES(s):
        cnl = ControlNetLoader.INPUT_TYPES()
        cnaa = ControlNetApplyAdvanced.INPUT_TYPES()
        suct = SetUnionControlNetType.INPUT_TYPES()

        return {
            "required": {
                "model": ('MODEL',),
                "positive": cnaa['required']['positive'],
                "negative": cnaa['required']['negative'],
                "image": cnaa['required']['image'],
                "f": cnl['required']['control_net_name'],
                "type": suct['required']['type'],
                "strength": cnaa['required']['strength'],
                "start": ('FLOAT', {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "end": ('FLOAT', {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
            },
            "optional": {
                "mask": ('MASK',),
                "vae": ('VAE',),
            }
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING')
    RETURN_NAMES = ('MODEL', 'POSITIVE', 'NEGATIVE')

    def execute(self, model, positive, negative, f, type, image, strength, start, end, mask=None, vae=None):
        g = GraphBuilder()
        g_cn = g.node('ControlNetLoader', control_net_name=f)

        if type != 'auto':
            g_cn = g.node('SetUnionControlNetType', control_net=g_cn.out(0), type=type)

        if mask is not None:
            g_masked = g.node(
                'ConditioningSetMask',
                conditioning = positive,
                strength       = 1.0,
                set_cond_are   = 'default'
            )

            g_combined = g.node(
                'ConditioningCombine',
                conditioning_1 = positive,
                conditioning_2 = g_masked.out(0)
            )

            o_strength = strength * 2 # sensible adjustment, may still require high base strength
            o_positive = g_combined.out(0)
        else:
            o_strength = strength
            o_positive = positive

        g_apply = g.node(
            'ControlNetApplyAdvanced',
            positive      = o_positive,
            negative      = negative,
            control_net   = g_cn.out(0),
            image         = image,
            strength      = o_strength,
            start_percent = start,
            end_percent   = end,
            vae           = vae,
        )

        ui = PreviewImage().save_images(image, 'UseControlNet')['ui']

        return {
            'ui': ui,
            'result': (model, g_apply.out(0), g_apply.out(1)),
            'expand': g.finalize()
        }


# --------------------------------------------------------------------------------------------------

class UseStyleModel(Personal):
    @classmethod
    def INPUT_TYPES(s):
        sml = StyleModelLoader.INPUT_TYPES()
        sma = StyleModelApply.INPUT_TYPES()
        cvl = CLIPVisionLoader.INPUT_TYPES()
        cve = CLIPVisionEncode.INPUT_TYPES()

        return {
            "required": {
                'model': sml['required']['style_model_name'],
                'positive': sma['required']['conditioning'],
                'image': cve['required']['image'],
                "crop": cve['required']['crop'],
                "strength": sma['required']['strength'],
                "effect": sma['required']['strength_type'],
            },
        }

    RETURN_TYPES = ("CONDITIONING",)

    def execute(self, model, positive, image, crop, strength, effect):
        g = GraphBuilder()

        g_style_loader = g.node(
            'StyleModelLoader',
            style_model_name = model
        )

        g_cv_loader = g.node(
            'CLIPVisionLoader',
            clip_name = get_model_name('clip_vision', 'sigclip_vision_patch14_384.safetensors')
        )

        g_encode = g.node(
            'CLIPVisionEncode',
            clip_vision = g_cv_loader.out(0),
            image       = image,
            crop        = crop
        )

        g_apply = g.node(
            'StyleModelApply',
            conditioning       = positive,
            style_model        = g_style_loader.out(0),
            clip_vision_output = g_encode.out(0),
            strength           = strength,
            strength_type      = effect
        )

        return {
            'result': (g_apply.out(0),),
            'expand': g.finalize()
        }


# --------------------------------------------------------------------------------------------------

class UseIPAdapter(Personal):

    @classmethod
    def INPUT_TYPES(s):
        ipaml = IPAdapterModelLoader.INPUT_TYPES()
        ipaa = IPAdapterAdvanced.INPUT_TYPES()
        cvl = CLIPVisionLoader.INPUT_TYPES()

        return {
            "required": {
                'model': ipaa['required']['model'],
                'clip': (folder_paths.get_filename_list("clip_vision"), { 'tooltip': "Maybe ViT-H 14 Laion2B s32B b79k" }),
                'ipa': (folder_paths.get_filename_list("ipadapter"),),
                'positive': ipaa['required']['image'],
                "weight": ipaa['required']['weight'],
                "effect": ipaa['required']['weight_type'],
                "combine": ipaa['required']['combine_embeds'],
                "start": ('FLOAT', {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
                "end": ('FLOAT', {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "round": 0.01}),
            },
            'optional': {
                'negative': ipaa['optional']['image_negative'],
                'mask': ipaa['optional']['attn_mask']
            }
        }

    RETURN_TYPES = ('MODEL',)

    def execute(self, model, clip, ipa, positive, weight, effect, combine, start, end, negative=None, mask=None):
        g = GraphBuilder()

        model_kind = get_model_kind(model)
        
        g_cv_loader = g.node('CLIPVisionLoader', clip_name=clip)
        g_ipa_loader = g.node('IPAdapterModelLoader', ipadapter_file=ipa)

        if mask is not None:
            mask = 1 - mask # mask hides, does not select

        g_apply = g.node(
            'IPAdapterAdvanced',
            model          = model,
            ipadapter      = g_ipa_loader.out(0),
            clip_vision    = g_cv_loader.out(0),
            start_at       = start,
            end_at         = end,
            weight         = weight,
            weight_type    = effect,
            image          = positive,
            image_negative = negative,
            attn_mask      = mask
        )

        return {
            'result': (g_apply.out(0),),
            'expand': g.finalize()
        }


# --------------------------------------------------------------------------------------------------

class UseInfiniteYou(Personal):
    @classmethod
    def INPUT_TYPES(s):
        ideml_types = IDEmbeddingModelLoader.INPUT_TYPES()
        efpi_types = ExtractFacePoseImage.INPUT_TYPES()
        inl_types = InfuseNetLoader.INPUT_TYPES()
        ina = InfuseNetApply.INPUT_TYPES()

        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "image_kps": ("IMAGE",),
                "cn": inl_types['required']['controlnet_name'],
                "proj_model": ideml_types['required']['image_proj_model_name'],
                "cn_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05, }),
                "proj_tokens": ideml_types['required']['image_proj_num_tokens'],
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "vae": ("VAE", ),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING')
    RETURN_NAMES = ('MODEL', 'POSITIVE', 'NEGATIVE')

    def execute(self, model, positive, negative, image, image_kps, cn, proj_model, cn_strength, proj_tokens,
        start, end, vae=None, mask=None):
        g = GraphBuilder()

        g_control_net = g.node(
            'InfuseNetLoader', 
            controlnet_name = cn
        )

        g_loader = g.node(
            'IDEmbeddingModelLoader',
            image_proj_model_name  = proj_model,
            image_proj_num_tokens  = proj_tokens,
            face_analysis_provider = 'CPU',
            face_analysis_det_size = 'AUTO'
        )

        height = image_kps.shape[1]
        width = image_kps.shape[2]

        g_extract_pose = g.node(
            'ExtractFacePoseImage',
            face_detector = g_loader.out(0),
            image         = image_kps,
            width         = width,
            height        = height
        )

        g_extract_embed = g.node(
            'ExtractIDEmbedding',
            face_detector    = g_loader.out(0),
            arcface_model    = g_loader.out(1),
            image_proj_model = g_loader.out(2),
            image            = image
        )

        g_apply = g.node(
            'InfuseNetApply',
            positive      = positive,
            negative      = negative,
            id_embedding  = g_extract_embed.out(0),
            control_net   = g_loader.out(0),
            image         = g_extract_pose.out(0),
            strength      = cn_strength,
            start_percent = start,
            end_percent   = end,
            vae           = vae,
            control_mask  = mask,
        )

        return {
            'result': (model, g_apply.out(0), g_apply.out(1)),
            'expand': g.finalize(),
        }


# --------------------------------------------------------------------------------------------------

class UseImage(Personal):

    @classmethod
    def INPUT_TYPES(s):
        dd = DifferentialDiffusion.INPUT_TYPES()
        imc = InpaintModelConditioning.INPUT_TYPES()

        return {
            "required": {
                "model": dd['required']['model'],
                "positive": imc['required']['positive'],
                "negative": imc['required']['negative'],
                "image": imc['required']['pixels'],
            },
            "optional": {
                "mask": imc['required']['mask'],
                "vae": imc['required']['vae'], # vae is not optional, but I want this order :)
            }
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING', 'LATENT')
    RETURN_NAMES = ('MODEL', 'POSITIVE', 'NEGATIVE', 'LATENT')

    def execute(self, model, positive, negative, image, mask=None, vae=None):
        g = GraphBuilder()

        if vae is None:
            raise Exception("Missing VAE input in UseImage node")

        if mask is None or torch.all(mask > 0.999):
            g_latent = g.node('VAEEncode', vae=vae, pixels=image)
            result = (model, positive, negative, g_latent.out(0))

        else:
            g_mask = g.node('GrowMask', mask=mask, expand=8, tapered_corners=True)
            g_mask = g.node('ImpactGaussianBlurMask', mask=g_mask.out(0), kernel_size=8, sigma=10)
            g_model = g.node('DifferentialDiffusion', model=model)

            g_cond = g.node(
                'InpaintModelConditioning',
                positive   = positive,
                negative   = negative,
                pixels     = image,
                vae        = vae,
                mask       = g_mask.out(0),
                noise_mask = True
            )

            result = (g_model.out(0), g_cond.out(0), g_cond.out(1), g_cond.out(2))

        return {
            'result': result,
            'expand': g.finalize()
        }

# --------------------------------------------------------------------------------------------------

class UseInstantID(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", ),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "image": ("IMAGE", ),
                "ip_weight": ("FLOAT", {"default": .4, "min": 0.0, "max": 3.0, "step": 0.05, }),
                "cn_strength": ("FLOAT", {"default": .8, "min": 0.0, "max": 10.0, "step": 0.05, }),
                "noise": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01, }),
                "embeds": (['average', 'norm average', 'concat'], {"default": 'average'}),
                "start": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
                "end": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, }),
            },
            "optional": {
                "image_kps": ("IMAGE",),
                "mask": ("MASK",),
            }
        }

    RETURN_TYPES = ('MODEL', 'CONDITIONING', 'CONDITIONING')
    RETURN_NAMES = ('MODEL', 'POSITIVE', 'NEGATIVE')

    def execute(
        self,
        model,
        image,
        positive, negative,
        ip_weight, cn_strength,
        noise,
        embeds,
        start, end,
        image_kps=None, mask=None,
    ):
        g = GraphBuilder()

        g_control_net = g.node('ControlNetLoader', control_net_name=get_model_name('controlnet', 'instantx_instantid'))
        g_instantid = g.node('InstantIDModelLoader', instantid_file=get_model_name('instantid', 'instantx_instantid'))
        g_insightface = g.node('InstantIDFaceAnalysis', provider='cpu')

        g_apply = g.node(
            'ApplyInstantIDAdvanced',
            instantid      = g_instantid.out(0),
            insightface    = g_insightface.out(0),
            control_net    = g_control_net.out(0),
            image          = image,
            model          = model,
            positive       = positive,
            negative       = negative,
            start_at       = start,
            end_at         = end,
            weight         = 0, # combined weight/strength param, superseded by ip_weight and cn_strength
            ip_weight      = ip_weight,
            cn_strength    = cn_strength,
            noise          = noise,
            image_kps      = image_kps,
            mask           = mask,
            combine_embeds = embeds
        )

        return {
            'result': (g_apply.out(0), g_apply.out(1), g_apply.out(2)),
            'expand': g.finalize()
        }

# --------------------------------------------------------------------------------------------------


class GenerateImage(Personal):
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        ks = KSampler.INPUT_TYPES()
        pag = PerturbedAttentionGuidance.INPUT_TYPES()
        si = SaveImage.INPUT_TYPES()
        vaed = VAEDecode.INPUT_TYPES()
        tcd = TCDModelSamplingDiscrete.INPUT_TYPES()

        samplers = ['dpmpp_2m', 'tcd', 'euler', 'dpmpp_sde', 'dpmpp_2m_sde', 'euler_ancestral']
        # schedulers = comfy.samplers.KSampler.SCHEDULERS

        return {
            "required": {
                "model": pag['required']['model'],
                "seed": ks['required']['seed'],
                "steps": ks['required']['steps'],
                "sampler": (samplers,),
                "positive": ks['required']['positive'],
                "negative": ks['required']['negative'],
                "latent": ks['required']['latent_image'],
                "cfg": ("FLOAT", {"default": 4.0, "min": 0.0, "max": 100.0, "step": 0.05, "round": 0.01}),
                "pag": pag['required']['scale'],
                "denoise": ks['required']['denoise'],
                "save": ('BOOLEAN', {'default': True}),
                "vae": vaed['required']['vae'],
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("IMAGE", "LATENT",)

    def execute(
        self,
        model,
        seed, steps, sampler,
        positive, negative, latent,
        cfg, pag, denoise,
        save,
        vae,
        prompt=None, extra_pnginfo=None,
    ):
        g = GraphBuilder()
        g_model = g.node('ForwardModel', model=model)

        if pag > 0:
            g_model = g.node('PerturbedAttentionGuidance', model=model, scale=pag)

        scheduler = {
            'dpmpp_2m': 'simple',
            'tcd': 'normal',
            'euler': 'normal',
            'dpmpp_sde': 'karras',
            'dpmpp_2m_sde': 'karras',
            'euler_ancestral': 'normal'
        }[sampler]

        eta = None

        if sampler == 'tcd':
            # Pick a good eta value, no need to be fancy:
            if steps >= 8: eta = 0.5
            elif steps >= 2: eta = 0.8
            else: eta = 1.0

            # Load the lora:
            g_lora = g.node(
                'LoraLoaderModelOnly',
                model          = g_model.out(0),
                lora_name      = get_model_kind(model) + '_lora_tcd.safetensors',
                strength_model = 1.0
            )

            # Patch the model and get the sampler conf:
            g_tcd = g.node(
                'TCDModelSamplingDiscrete',
                model     = g_lora.out(0),
                steps     = steps,
                scheduler = scheduler,
                denoise   = denoise,
                eta       = eta
            )

            o_model = g_tcd.out(0)
            o_sampler = g_tcd.out(1)
            o_sigmas = g_tcd.out(2)

        else:
            g_sampler = g.node(
                'KSamplerSelect',
                sampler_name = sampler
            )

            g_scheduler = g.node(
                'BasicScheduler',
                model = g_model.out(0),
                steps = steps,
                scheduler = scheduler,
                denoise = denoise
            )

            o_model = g_model.out(0)
            o_sampler = g_sampler.out(0)
            o_sigmas = g_scheduler.out(0)

        # Sample:
        g_latent = g.node(
            'SamplerCustom',
            model        = o_model,
            add_noise    = True,
            noise_seed   = seed,
            cfg          = cfg,
            positive     = positive,
            negative     = negative,
            sampler      = o_sampler,
            sigmas       = o_sigmas,
            latent_image = latent
        )

        g_decode = g.node(
            'VAEDecode',
            vae=vae,
            samples=g_latent.out(0)
        )

        if save:
            prefix = '_'.join(filter(lambda it: it is not None, [
                f"{int(time.time())}",
                get_model_kind(model),
                f"i{steps}",
                f"{sampler.replace('_', '')}",
                f"{scheduler.replace('_', '')[:4]}",
                f"cfg{cfg if cfg % 1 else int(cfg)}",
                f"pag{pag if pag % 1 else int(pag)}",
                f"eta{eta}" if sampler == 'tcd' else None,
                f"n{denoise if denoise % 1 else int(denoise)}",
                f"s{seed}"
            ]))

            g.node(
                'SaveImage',
                images          = g_decode.out(0),
                filename_prefix = prefix,
                prompt          = prompt,
                extra_pnginfo   = extra_pnginfo
            )
        else:
            g.node(
                'PreviewImage',
                images          = g_decode.out(0),
                prompt          = prompt,
                extra_pnginfo   = extra_pnginfo
            )

        return {
            'result': (g_decode.out(0), g_latent.out(0)),
            'expand': g.finalize()
        }

# --------------------------------------------------------------------------------------------------


class UpscaleImage(Personal):
    OUTPUT_NODE = True

    @classmethod
    def INPUT_TYPES(s):
        uml = UpscaleModelLoader.INPUT_TYPES()
        iuwm = ImageUpscaleWithModel.INPUT_TYPES()

        return {
            "required": {
                "f": uml['required']['model_name'],
                "image": iuwm['required']['image'],
                "save": ('BOOLEAN', {'default': True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    RETURN_TYPES = ('IMAGE',)

    def execute(self, f, image, save, prompt=None, extra_pnginfo=None):
        g = GraphBuilder()

        g_model = g.node('UpscaleModelLoader', model_name=f)
        g_apply = g.node('ImageUpscaleWithModel', upscale_model=g_model.out(0), image=image)

        if save:
            g.node(
                'SaveImage',
                images          = g_apply.out(0),
                filename_prefix = f'{int(time.time())}_upscale_{f}',
                prompt          = prompt,
                extra_pnginfo   = extra_pnginfo
            )
        else:
            g.node(
                'PreviewImage',
                images          = g_apply.out(0),
                filename_prefix = f'upscale_{f}',
                prompt          = prompt,
                extra_pnginfo   = extra_pnginfo
            )

        return {
            'result': (g_apply.out(0),),
            'expand': g.finalize()
        }

# --------------------------------------------------------------------------------------------------


class OwlDetector(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ('IMAGE',),
                "model": (['google/owlv2-large-patch14-ensemble'],),
                "size": ('INT', {'default': 512}),
                "threshold": ('FLOAT', {'default':0.1}),
                "prompt": ('STRING',),
            }
        }

    RETURN_TYPES = ('IMAGE',)
    # RETURN_TYPES = ('IMAGE', 'IMAGE',)
    # RETURN_NAMES = ('BEST', 'SECOND')

    def execute(self, image, model, size, threshold, prompt):
        from transformers import Owlv2Processor, Owlv2ForObjectDetection

        processor = Owlv2Processor.from_pretrained(model, use_fast=True, do_rescale=False)
        detector = Owlv2ForObjectDetection.from_pretrained(model)

        texts = [prompt.split(',')] # 1 batch of many terms

        inputs = processor(text=texts, images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = detector(**inputs)

        results = processor.post_process_object_detection(
            outputs,
            threshold    = threshold,
            target_sizes = torch.tensor([[ image.shape[1], image.shape[2] ]])
        )[0]

        if len(results['boxes']) < 1:
            return None # TODO

        items = sorted(
            zip(results['boxes'], results['labels'], results['scores']),
            key     = lambda item: item[2],
            reverse = True
        )

        box, label, score = items[0]
        return self._crop(image, box, size)

    def _crop(self, image, box, size):
        _, height, width, _ = image.shape

        # Get center of bounding box
        x1, y1, x2, y2 = box
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)

        # Find vertical bounds:
        top = max(0, cy - size // 2)
        bottom = min(height, top + size)
        
        if bottom == height:
            top = height - size

        # Find horizontal bounds:
        left = max(0, cx - size // 2)
        right = min(width, left + size)

        if right == width:
            left = width - size

        # Croppity crop:
        cropped = image[0, top:bottom, left:right, :]
        return (cropped.unsqueeze(0),)

# --------------------------------------------------------------------------------------------------

class SliceImageBatch(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                'image': ('IMAGE',),
                'start': ('INT', {'default': 0, 'tooltip': "Start index, inclusive (negative counts from the end)"}),
                'end': ('INT', {'default': -1, 'min': -1000, 'max': 1000, 'tooltip': "End index, inclusive (negative counts from the end)"}),
            }
        }

    RETURN_TYPES = ('IMAGE',)

    def execute(self, image, start, end):
        if end == -1:
            image = image[start:]
        else:
            image = image[start:end+1]

        return (image,)


class SliceMaskBatch(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                'mask': ('MASK',),
                'start': ('INT', {'default': 0, 'tooltip': "Start index, inclusive (negative counts from the end)"}),
                'end': ('INT', {'default': -1, 'min': -1000, 'max': 1000, 'tooltip': "End index, inclusive (negative counts from the end)"}),
            }
        }

    RETURN_TYPES = ('MASK',)

    def execute(self, mask, start, end):
        if end == -1:
            mask = mask[start:]
        else:
            mask = mask[start:end+1]

        return (mask,)


class MaskBatch(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask1": ("MASK",),
                "mask2": ("MASK",),
            }
        }

    RETURN_TYPES = ("MASK",)

    def execute(self, mask1, mask2):
        concatenated = torch.cat((mask1, mask2), dim=0)
        return (concatenated,)


class RepeatMaskBatch(Personal):
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
                "amount": ("INT", {"default": 1, "min": 1, "max": 4096}),
            }
        }

    RETURN_TYPES = ("MASK",)

    def execute(self, mask, amount):
        repeated = mask.repeat((amount, 1,1))
        return (repeated,)

# --------------------------------------------------------------------------------------------------


def get_model_name(kind, pat):
    names = folder_paths.get_filename_list(kind)
    match = None

    if isinstance(pat, str):
        match = next(filter(lambda n: n.startswith(pat), names), None)
    else:
        match = next(filter(pat.match, names), None)

    if not match:
        raise ValueError(f"No match for model {kind}/{pat}")

    return match


def get_model_kind(model):
    return type(model.model.model_config).__name__.lower()


def create_forward_node_class(vtype, **kwargs):
    vtype = vtype.upper()

    class cls:
        CATEGORY = "personal"
        FUNCTION = "execute"

        @classmethod
        def INPUT_TYPES(cls):
            return {"optional": {vtype.lower(): (vtype, {'multiline': True, **kwargs})}}

        RETURN_TYPES = (vtype,)
        RETURN_NAMES = (vtype,)

        def execute(self, **kwargs):
            return tuple(kwargs.values())

    cls.__name__ = "Forward" + vtype.title()
    return cls


vtypes = (
    'MODEL',
    'VAE',
    'CLIP',
    'CONDITIONING',
    'LATENT',
    'IMAGE',
    'MASK',
    'INT',
    'STRING',
    'FLOAT',
    'BOOLEAN'
)

NODE_CLASS_MAPPINGS = {
    'UseCheckpoint': UseCheckpoint,
    'UseControlNet': UseControlNet,
    'UseLora': UseLora,
    'UseIPAdapter': UseIPAdapter,
    'UseStyleModel': UseStyleModel,
    'UseImage': UseImage,
    'UseInstantID': UseInstantID,
    'UseInfiniteYou': UseInfiniteYou,
    'GenerateImage': GenerateImage,
    'UpscaleImage': UpscaleImage,
    'SliceImageBatch': SliceImageBatch,
    'SliceMaskBatch': SliceMaskBatch,
    'RepeatMaskBatch': RepeatMaskBatch,
    'MaskBatch': MaskBatch,
    'OwlDetector': OwlDetector,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    'UseCheckpoint': "Use Checkpoint",
    'UseControlNet': "Use ControlNet",
    'UseLora': "Use Lora",
    'UseIPAdapter': "Use IPAdapter",
    'UseStyleModel': "Use Style Model",
    'UseImage': "Use Image",
    'UseInstantID': "Use InstantID",
    'UseInfiniteYou': "Use InfiniteYou",
    'GenerateImage': "Generate Image",
    'UpscaleImage': "Upscale Image",
    'CLIPTextEncodeVar': "CLIP Text Encode (Var)",
    'FillMask': "Fill Mask",
    'SliceImageBatch': "Slice Image Batch",
    'SliceMaskBatch': "Slice Mask Batch",
    'RepeatMaskBatch': "Repeat Mask Batch",
    'MaskBatch': "Batch Masks",
    'OwlDetector': "Owl Detector",
}

for vtype in vtypes:
    cls = create_forward_node_class(vtype)

    NODE_CLASS_MAPPINGS[cls.__name__] = cls
    NODE_DISPLAY_NAME_MAPPINGS[cls.__name__] = "Forward " + vtype.title()



WEB_DIRECTORY = './js'

__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']

import argparse

import numpy as np


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--weight-decay', default=1e-4, type=float)
    parser.add_argument('--lr-drop', default=10, type=int)
    parser.add_argument('--clip-max-norm', default=0.1, type=float)
    parser.add_argument('--backbone', default='resnet50', type=str)
    parser.add_argument('--dilation', action='store_true')
    parser.add_argument('--amp', action='store_true')
    parser.add_argument(
        '--position-embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
    )
    parser.add_argument('--repr-dim', default=512, type=int)
    parser.add_argument('--hidden-dim', default=256, type=int)
    parser.add_argument('--enc-layers', default=6, type=int)
    parser.add_argument('--dec-layers', default=6, type=int)
    parser.add_argument('--dim-feedforward', default=2048, type=int)
    parser.add_argument('--dropout', default=0.1, type=float)
    parser.add_argument('--nheads', default=8, type=int)
    parser.add_argument('--num-queries', default=100, type=int)
    parser.add_argument('--pre-norm', action='store_true')

    parser.add_argument('--no-aux-loss', dest='aux_loss', action='store_false')
    parser.add_argument('--set-cost-class', default=1, type=float)
    parser.add_argument('--set-cost-bbox', default=5, type=float)
    parser.add_argument('--set-cost-giou', default=2, type=float)
    parser.add_argument('--bbox-loss-coef', default=5, type=float)
    parser.add_argument('--giou-loss-coef', default=2, type=float)
    parser.add_argument(
        '--eos-coef',
        default=0.1,
        type=float,
        help='Relative classification weight of the no-object class',
    )

    # Training parameters
    parser.add_argument('--cache', action='store_true')
    parser.add_argument('--box-score-thresh', default=0.2, type=float)
    parser.add_argument('--fg-iou-thresh', default=0.5, type=float)

    # [VG frozen literal text]
    # Keep the existing prompt + MLP composition as the default. The literal-cache
    # path changes only VG text representation for a controlled ablation.
    parser.add_argument(
        '--vg-text-mode',
        default='compositional',
        choices=[
            'compositional',
            'literal_cache',
            'online_literal',
        ],
        help=(
            'VG relation text representation. '
            'compositional preserves the trainable prompt + MLP path; '
            'literal_cache uses frozen CLIP S-P-O embeddings.'
            'online_literal encodes trainable literal S-P-O prompts '
            'during every forward pass.'
        ),
    )
    parser.add_argument(
        '--vg-literal-cache',
        default='',
        type=str,
        help=(
            'Path to the external frozen VG literal S-P-O cache. '
            'Required only with --vg-text-mode literal_cache.'
        ),
    )
    # [VG online literal text]
    # Number of literal S-P-O prompts processed by the CLIP text transformer
    # at once. This bounds activation memory without changing candidate order.
    parser.add_argument(
        '--text-prompt-batch-size',
        default=256,
        type=int,
        help=(
            'CLIP text chunk size for --vg-text-mode online_literal.'
        ),
    )

    parser.add_argument('--min-instances', default=3, type=int)
    parser.add_argument('--max-instances', default=15, type=int)

    # CLIP ResNet arguments retained from the original implementation.
    # parser.add_argument('--clip_dir', default='./checkpoints/pretrained_clip/RN50.pt', type=str)
    # parser.add_argument('--clip_visual_layers', default=[3, 4, 6, 3], type=list)
    # parser.add_argument('--clip_visual_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_visual_input_resolution', default=1344, type=int)
    # parser.add_argument('--clip_visual_width', default=64, type=int)
    # parser.add_argument('--clip_visual_patch_size', default=64, type=int)
    # parser.add_argument('--clip_text_output_dim', default=1024, type=int)
    # parser.add_argument('--clip_text_transformer_width', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers', default=12, type=int)
    # parser.add_argument('--clip_text_context_length', default=13, type=int)

    # CLIP ViT-L/14@336px
    parser.add_argument('--clip_visual_layers_vit', default=24, type=list)
    parser.add_argument('--clip_visual_output_dim_vit', default=768, type=int)
    parser.add_argument('--clip_visual_input_resolution_vit', default=336, type=int)
    parser.add_argument('--clip_visual_width_vit', default=1024, type=int)
    parser.add_argument('--clip_visual_patch_size_vit', default=14, type=int)
    parser.add_argument('--clip_text_transformer_width_vit', default=768, type=int)
    parser.add_argument('--clip_text_transformer_heads_vit', default=12, type=int)
    parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)

    # CLIP ViT-B/16 alternatives retained from the original implementation.
    # parser.add_argument('--clip_visual_layers_vit', default=12, type=list)
    # parser.add_argument('--clip_visual_output_dim_vit', default=512, type=int)
    # parser.add_argument('--clip_visual_input_resolution_vit', default=224, type=int)
    # parser.add_argument('--clip_visual_width_vit', default=768, type=int)
    # parser.add_argument('--clip_visual_patch_size_vit', default=16, type=int)
    # parser.add_argument('--clip_text_transformer_width_vit', default=512, type=int)
    # parser.add_argument('--clip_text_transformer_heads_vit', default=8, type=int)
    # parser.add_argument('--clip_text_transformer_layers_vit', default=12, type=int)
    parser.add_argument('--clip_text_context_length_vit', default=77, type=int)

    # 0: dropout (random mask); 1: no feature mask.
    parser.add_argument('--feat_mask_type', type=int, default=0)
    parser.add_argument(
        '--repeat_factor_sampling',
        default=False,
        type=lambda x: str(x).lower() == 'true',
        help=(
            'Apply repeat factor sampling to increase the rate at which '
            'tail categories are observed.'
        ),
    )
    parser.add_argument('--dataset_file', default='coco')

    # Deformable DETR arguments
    parser.add_argument(
        '--d_detr',
        default=False,
        type=lambda x: str(x).lower() == 'true',
    )
    parser.add_argument('--lr_backbone', default=2e-5, type=float)
    parser.add_argument(
        '--masks',
        action='store_true',
        help='Train segmentation head if the flag is provided',
    )
    parser.add_argument('--with_box_refine', default=False, action='store_true')
    parser.add_argument('--two_stage', default=False, action='store_true')
    parser.add_argument(
        '--frozen_weights',
        type=str,
        default=None,
        help=(
            'Path to the pretrained model. If set, only the mask head '
            'will be trained.'
        ),
    )
    parser.add_argument(
        '--position_embedding',
        default='sine',
        type=str,
        choices=('sine', 'learned'),
        help='Type of positional embedding to use on top of the image features',
    )
    parser.add_argument(
        '--position_embedding_scale',
        default=2 * np.pi,
        type=float,
        help='position / size * scale',
    )
    parser.add_argument(
        '--num_feature_levels',
        default=4,
        type=int,
        help='number of feature levels',
    )
    parser.add_argument('--dec_n_points', default=4, type=int)
    parser.add_argument('--enc_n_points', default=4, type=int)
    parser.add_argument('--mask_loss_coef', default=1, type=float)
    parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--cls_loss_coef', default=2, type=float)
    parser.add_argument('--focal_alpha', default=0.25, type=float)

    parser.add_argument('--lr-head', default=1e-3, type=float)
    parser.add_argument('--lr-vit', default=1e-3, type=float)

    parser.add_argument('--zs', action='store_true')
    parser.add_argument(
        '--zs_type',
        type=str,
        default='rare_first',
        choices=[
            'rare_first',
            'non_rare_first',
            'unseen_verb',
            'uc0',
            'uc1',
            'uc2',
            'uc3',
            'uc4',
            'unseen_object',
        ],
    )
        # [VG open-vocabulary relation split]
    # This is separate from the original HICO zero-shot configurations.
    parser.add_argument(
        "--vg-ovr",
        action="store_true",
        help=(
            "Use the official OvSGTR VG150 relation split: "
            "train on 35 base predicates and evaluate all 50 predicates "
            "on split_GLIPunseen."
        ),
    )

    parser.add_argument('--dataset', default='hicodet', type=str)
    parser.add_argument(
        '--partitions',
        nargs='+',
        default=['train2015', 'test2015'],
        type=str,
    )
    parser.add_argument('--num_classes', type=int, default=117)
    parser.add_argument('--data-root', default='./hicodet')

    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch-size', default=8, type=int)
    parser.add_argument('--num-workers', default=2, type=int)
    parser.add_argument('--eval', action='store_true')

    parser.add_argument(
        '--clip_dir_vit',
        default='./checkpoints/pretrained_clip/ViT-B-16.pt',
        type=str,
    )
    parser.add_argument('--pretrained', default='', help='Path to a pretrained detector')
    parser.add_argument(
        '--egtr-detector-dir',
        default='',
        type=str,
        help=(
            'Directory containing the official EGTR VG detector '
            'config.json and pytorch_model.bin.'
        ),
    )
    parser.add_argument('--resume', default='', help='Resume from a model')
    parser.add_argument('--output-dir', default='checkpoints')

    parser.add_argument('--use_hotoken', action='store_true')
    parser.add_argument('--use_prior', action='store_true')
    parser.add_argument('--use_exp', action='store_true')

    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--gamma', default=0.2, type=float)
    parser.add_argument('--hyper_lambda', type=float, default=2.8)

    # Adapter
    parser.add_argument('--use_insadapter', action='store_true')
    parser.add_argument('--adapter_num_layers', type=int, default=1)
    parser.add_argument('--adapt_dim', default=32, type=int)
    parser.add_argument('--adapter_alpha', default=1.0, type=float)
    parser.add_argument(
        '--adapter_pos',
        type=str,
        default='all',
        choices=['all', 'front', 'end', 'random', 'last', '03', '47', '811'],
    )
    parser.add_argument(
        '--adapter_scalar',
        default='learnable_scalar',
        type=str,
    )

    # Prompt learning
    parser.add_argument('--use_prompt', action='store_true')
    parser.add_argument('--N_CTX', type=int, default=24)
    parser.add_argument('--CSC', action='store_true')
    parser.add_argument('--CTX_INIT', type=str, default='')
    parser.add_argument('--CLASS_TOKEN_POSITION', type=str, default='end')
    # Miscellaneous
    parser.add_argument('--job_id', default=1985, type=int)
    parser.add_argument('--vis', action='store_true')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument(
        '--debug-samples',
        default=20,
        type=int,
        help=(
            'Number of train and test images exposed by debug_main.py. '
            'This changes only debug run scale, not the VG split.'
        ),
    )
    parser.add_argument('--seed', default=66, type=int)
    parser.add_argument(
        '--device',
        default='cuda',
        help='device to use for training / testing',
    )
    parser.add_argument('--port', default='7894', type=str)
    parser.add_argument('--print-interval', default=100, type=int)
    parser.add_argument('--world-size', default=1, type=int)

    return parser.parse_args()

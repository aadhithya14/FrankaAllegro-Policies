import hydra 

from torch.nn.parallel import DistributedDataParallel as DDP

from .byol import BYOLLearner
from .vicreg import VICRegLearner
from .behavior_cloning import ImageTactileBC
from .bet import BETLearner
from .bc_gmm import BCGMM
from .mocov3 import MOCOLearner
from .simclr import SIMCLRLearner
from .temporal_ssl import TemporalSSLLearner
from .dynamics import DynamicsLearner

from franka_allegro.models.self_supervised_pretraining.dynamics import resnet18jeff
from franka_allegro.utils import *
from franka_allegro.models import  *

def init_learner(cfg, device, rank=0):
    if cfg.learner_type == 'bc':
        return init_bc(cfg, device, rank)
    elif cfg.learner_type == 'bc_gmm':
        return init_bc_gmm(cfg, device, rank)
    elif 'tactile' in cfg.learner_type:
        return init_tactile_byol(
            cfg,
            device, 
            rank,
            aug_stat_multiplier=cfg.learner.aug_stat_multiplier,
            byol_in_channels=cfg.learner.byol_in_channels,
            byol_hidden_layer=cfg.learner.byol_hidden_layer
        )
    elif cfg.learner_type == 'image_byol':
        return init_image_byol(cfg, device, rank)
    elif 'vicreg' in cfg.learner_type:
        return init_vicreg(
            cfg,
            device,
            rank,
            sim_coef=cfg.learner.sim_coef,
            std_coef=cfg.learner.std_coef,
            cov_coef=cfg.learner.cov_coef)
    
    elif cfg.learner_type == 'bet':
        return init_bet_learner(
            cfg,
            device
        )
    elif cfg.learner_type == 'temporal_ssl':
        return init_temporal_learner(
            cfg,
            device,
            rank
        )

    elif cfg.learner_type == 'dynamics':
        return init_dynamics_learner(
            cfg,
            device,
            rank
        )

    return None



def init_tactile_moco(cfg, device, rank):
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)
    # Change the last layer
    encoder.classifier = create_fc(
        input_dim = cfg.learner.last_layer.input_dim,
        output_size = cfg.encoder.out_dim,
        hidden_size = cfg.learner.last_layer.hidden_size,
        use_batchnorm = cfg.learner.last_layer.use_batchnorm,
        is_moco = True
    )

    # Initialize the encoders and the augmentations
    momentum_encoder = copy(encoder)

    predictor = hydra.utils.instantiate(
        cfg.learner.predictor,
        input_dim = cfg.encoder.out_dim
    )
    first_augmentation_function, second_augmentation_function = get_moco_augmentations(
        mean_tensor = TACTILE_IMAGE_MEANS, 
        std_tensor = TACTILE_IMAGE_STDS
    )

    # Initialize the wrapper 
    moco_wrapper = MoCo (
        base_encoder = encoder,
        momentum_encoder = momentum_encoder,
        predictor = predictor,
        first_augment_fn = first_augmentation_function,
        sec_augment_fn = second_augmentation_function,
        temperature = cfg.learner.temperature,
        device = device
    )

    # Initialize the optimizer 
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    momentum_encoder = DDP(momentum_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    predictor = DDP(predictor, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = moco_wrapper.parameters())


    # Initialize the learner
    learner = MOCOLearner(
        wrapper = moco_wrapper, 
        optimizer = optimizer,
        momentum = cfg.learner.momentum,
        total_epochs = cfg.train_epochs
    )
    learner.to(device)

    return learner

def init_tactile_simclr(cfg, device, rank):
    # Initialize the augmentations and models to be used 
    augmentation_function = get_simclr_augmentation(
        color_jitter_const = cfg.learner.augmentation.color_jitter_const,
        mean_tensor = TACTILE_IMAGE_MEANS, 
        std_tensor =  TACTILE_IMAGE_STDS
    )

    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    projector = hydra.utils.instantiate(
        input_dim = cfg.encoder.out_dim
    )
    
    # Initialize the SimCLR loss wrapper
    simclr_wrapper = SimCLR(
        encoder = encoder, 
        projector = projector,
        augment_fn = augmentation_function,
        temperature = cfg.learner.temperature,
        device = device
    )

    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    projector = DDP(projector, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = simclr_wrapper.parameters())
    
    # Initialize the learner
    learner = SIMCLRLearner(
        simclr_wrapper = simclr_wrapper,
        optimizer = optimizer
    )
    learner.to(device)

    return learner

def init_bet_learner(cfg, device):
    bet_model = hydra.utils.instantiate(cfg.learner.model).to(device)

    optimizer = bet_model.configure_optimizers(
        weight_decay=cfg.learner.optim.weight_decay,
        learning_rate=cfg.learner.optim.lr,
        betas=cfg.learner.optim.betas
    )

    learner = BETLearner(
        bet_model = bet_model,
        optimizer = optimizer
    )
    learner.to(device)

    return learner

def init_vicreg(cfg, device, rank, sim_coef, std_coef, cov_coef):
    if 'tactile' in cfg.learner_type:
        augment_fn = get_tactile_augmentations(
            img_means = TACTILE_IMAGE_MEANS,
            img_stds = TACTILE_IMAGE_STDS,
            img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
        )
    else:
        augment_fn = get_vision_augmentations(
            img_means = VISION_IMAGE_MEANS,
            img_stds = VISION_IMAGE_STDS
        )

    backbone = hydra.utils.instantiate(cfg.encoder).to(device)

    # Initialize the vicreg projector
    projector = create_fc(
        input_dim = cfg.encoder.out_dim,
        output_dim = 8192,
        hidden_dims = [8192],
        use_batchnorm = True
    )

    # Initialize the vicreg wrapper 
    vicreg_wrapper = VICReg(
        backbone = backbone,
        projector = projector, 
        augment_fn = augment_fn, 
        sim_coef = sim_coef, 
        std_coef = std_coef, 
        cov_coef = cov_coef,
        device = device
    )

    if cfg.distributed:
        backbone = DDP(backbone, device_ids=[rank], output_device=rank, broadcast_buffers=False)
        projector = DDP(projector, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = vicreg_wrapper.parameters())

    learner = VICRegLearner(
        vicreg_wrapper = vicreg_wrapper, 
        optimizer = optimizer
    )                 
    learner.to(device)          

    return learner

def init_tactile_byol(cfg, device, rank, aug_stat_multiplier=1, byol_in_channels=3, byol_hidden_layer=-2):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_tactile_augmentations(
        img_means = TACTILE_IMAGE_MEANS*aug_stat_multiplier,
        img_stds = TACTILE_IMAGE_STDS*aug_stat_multiplier,
        img_size = (cfg.tactile_image_size, cfg.tactile_image_size)
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.tactile_image_size,
        augment_fn = augment_fn,
        hidden_layer = byol_hidden_layer,
        in_channels = byol_in_channels
    ).to(device)
    encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'tactile'
    )

    learner.to(device)

    return learner



def init_image_byol(cfg, device, rank):
    # Start the encoder
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS
    )
    # Initialize the byol wrapper
    byol = BYOL(
        net = encoder,
        image_size = cfg.vision_image_size,
        augment_fn = augment_fn
    ).to(device)
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = byol.parameters())
    
    # Initialize the agent
    learner = BYOLLearner(
        byol = byol,
        optimizer = optimizer,
        byol_type = 'image'
    )

    learner.to(device)

    return learner

def init_temporal_learner(cfg, device, rank):
    encoder = hydra.utils.instantiate(cfg.encoder.encoder).to(device)
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    linear_layer = hydra.utils.instantiate(cfg.encoder.linear_layer).to(device)
    if cfg.distributed:
        linear_layer = DDP(linear_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(encoder.parameters()) + list(linear_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=optim_params)

    learner = TemporalSSLLearner(
        optimizer = optimizer,
        repr_loss_fn = cfg.learner.repr_loss_fn,
        joint_diff_loss_fn = cfg.learner.joint_diff_loss_fn,
        encoder = encoder,
        linear_layer = linear_layer,
        joint_diff_scale_factor = cfg.learner.joint_diff_scale_factor,
        total_loss_type = cfg.learner.total_loss_type
    )
    learner.to(device)

    return learner

def init_bc(cfg, device, rank):
    image_encoder = hydra.utils.instantiate(cfg.encoder.image_encoder).to(device)
    if cfg.distributed:
        image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    tactile_encoder = hydra.utils.instantiate(cfg.encoder.tactile_encoder).to(device)
    if cfg.distributed:
        tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    last_layer = hydra.utils.instantiate(cfg.encoder.last_layer).to(device)
    if cfg.distributed:
        last_layer = DDP(last_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(image_encoder.parameters()) + list(tactile_encoder.parameters()) + list(last_layer.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    learner = ImageTactileBC(
        image_encoder = image_encoder, 
        tactile_encoder = tactile_encoder,
        last_layer = last_layer,
        optimizer = optimizer,
        loss_fn = cfg.learner.loss_fn,
        representation_type = cfg.learner.representation_type,
        freeze_encoders = cfg.learner.freeze_encoders
    )
    learner.to(device) 
    
    return learner

def init_bc_gmm(cfg, device, rank):
    # For this model we'll use already trained image and tactile encoders
    # Initialize image encoder
    _, image_encoder, _ = init_encoder_info(
        device, out_dir = cfg.learner.image_out_dir, encoder_type='image', view_num=cfg.learner.view_num
    ) # These are passed to the gpu device in load_model
    image_encoder = DDP(image_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the tactile encoder 
    _, tactile_encoder, _ = init_encoder_info(
        device, out_dir = cfg.learner.tactile_out_dir, encoder_type='tactile', view_num=cfg.learner.view_num
    ) # These are passed to the gpu device in load_model
    tactile_encoder = DDP(tactile_encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    # Initialize the gmm layer to guess the logits, mu and sigmas
    gmm_layer = hydra.utils.instantiate(
        cfg.learner.gmm_layer
    ).to(device)
    gmm_layer = DDP(gmm_layer, device_ids=[rank], output_device=rank, broadcast_buffers=False)

    optim_params = list(gmm_layer.parameters())
    if not cfg.learner.freeze_encoders:
        optim_params += list(image_encoder.parameters()) + list(tactile_encoder.parameters())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params = optim_params)

    learner = BCGMM(
        image_encoder = image_encoder,
        tactile_encoder = tactile_encoder,
        last_layer = gmm_layer,
        optimizer = optimizer,
        representation_type = cfg.learner.representation_type,
        freeze_encoders = cfg.learner.freeze_encoders
    )
    learner.to(device)

    return learner

def init_dynamics_learner(cfg,device,rank):
    encoder = hydra.utils.instantiate(cfg.encoder).to(device)

    dynamics_net = resnet18jeff(
    ).to(device)

    augment_fn = get_vision_augmentations(
        img_means = VISION_IMAGE_MEANS,
        img_stds = VISION_IMAGE_STDS
    )

     # Initialize the dynamics learning wrapper
    if cfg.distributed:
        encoder = DDP(encoder, device_ids=[rank], output_device=rank, broadcast_buffers=False)
    
    # Initialize the optimizer 
    optimizer = hydra.utils.instantiate(cfg.optimizer,
                                        params = dynamics_net.parameters())
    
    
    # Initialize the agent
    learner = DynamicsLearner(
        dynamics_learner = dynamics_net,
        optimizer = optimizer
    )

    learner.to(device)

    return learner
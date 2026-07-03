from pathlib import Path
import time
import lovely_tensors as lt
from datetime import datetime
import copy
import torch.optim as optim
import torch.nn.functional as F
import matplotlib as mpl
import matplotlib.cm as cm
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
import textwrap
from ruamel.yaml import YAML
from io import StringIO

from src.config.conf import Conf
from src.models.posenet.simple_pose_cnn import SimplePoseCNN
from src.models.posenet.resnet_pose_cnn import ResNetPoseCNN 
from src.models.monodepth2.monodepth2 import MonoDepth2
from src.models.hyenadepth.hyenadepth import HyenaDepth
from src.models.layers import *
from src.datasets.kitti_dataset import KITTIRAWDataset
from src.losses.loss import *
from src.utils import *
from data.kitti.kitti_utils.kitti_utils import *

class Trainer:
    def __init__(self, conf):
        self.conf = conf
        self.log_path = os.path.join(self.conf['tensorboard_path'], 'train')
        self.log_path = os.path.join(self.log_path, self.conf['model_name'])
        self.log_path = os.path.join(self.log_path, datetime.now().strftime("%Y%m%d-%H%M%S"))

        self.models = {}
        self.parameters_to_train = []

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.num_scales = len(self.conf['loss_scales']) 
        self.num_input_frames = len(self.conf['frame_ids_training']) # default=[0, -1, 1] => num_input_frames=3
        self.num_pose_frames = 2 if self.conf['pose_model_input'] == "pairs" else self.num_input_frames # default=2
        self.min_depth = self.conf['min_depth']
        self.max_depth = self.conf['max_depth']

        if self.conf['model_name'] == "hyena":
            self.models["depth_model"] = HyenaDepth(pretrained=self.conf['hyena']['pretrained'], scales=self.conf['hyena']['scales'], decoder_upsample=self.conf['hyena']['decoder_upsample'], disp_upsample=self.conf['hyena']['disp_upsample'], naf_kwargs=self.conf['hyena']['naf'])
            self.models["depth_model"].from_pretrained(encoder_weights_path=self.conf['hyena']['encoder_weights_path'], decoder_weights_path=self.conf['hyena']['decoder_weights_path'], weights_path=self.conf['hyena'].get('weights_path'), device=self.device) if self.conf['load_pretrained_depth_model'] else None
        elif self.conf['model_name'] == "monodepth2":
            self.models["depth_model"] = MonoDepth2(num_layers=self.conf['monodepth2']['num_layers'], pretrained=self.conf['monodepth2']['pretrained'], scales=self.conf['monodepth2']['scales'])
            self.models["depth_model"].from_pretrained(encoder_weights_path=self.conf['monodepth2']['encoder_weights_path'], decoder_weights_path=self.conf['monodepth2']['decoder_weights_path'], device=self.device) if self.conf['load_pretrained_depth_model'] else None
        else:
            print("Model not recognized!")
            exit()
        self.models["depth_model"] = self.models["depth_model"].to(self.device)
        self.parameters_to_train += list(self.models["depth_model"].parameters())

        # Prepare pose model 
        if self.conf['pose_model_type']=="simple_pose_cnn":
            self.models["pose_model"] = SimplePoseCNN(self.num_pose_frames)
            self.models["pose_model"].from_pretrained(weights_path=self.conf['simple_pose_cnn']['weights_path'], device=self.device) if self.conf['load_pretrained_pose_model'] else None
        elif self.conf['pose_model_type']=="resnet_pose_cnn":
            self.models["pose_model"] = ResNetPoseCNN(self.conf['resnet_pose_cnn']['num_layers'], self.conf['resnet_pose_cnn']['pretrained'], self.conf['resnet_pose_cnn']['num_input_images'], self.conf['resnet_pose_cnn']['num_input_features'], self.conf['resnet_pose_cnn']['num_frames_to_predict_for'])
            self.models["pose_model"].from_pretrained(weights_path=self.conf['resnet_pose_cnn']['weights_path'], device=self.device) if self.conf['load_pretrained_pose_model'] else None
        else:
            print("Pose model type not recognized!")
            exit()
        self.models["pose_model"] = self.models["pose_model"].to(self.device)
        self.parameters_to_train += list(self.models["pose_model"].parameters())

        print("Training model named:\n  ", self.conf['model_name'])
        print("Models and tensorboard events files are saved to:\n  ", self.conf['tensorboard_path'])
        print("Training is using:\n  ", self.device)
        print("Using split:\n  ", self.conf['training_split'])

        # Preparing data for training
        self.dataset = KITTIRAWDataset

        data_path = Path(self.conf["data_path"])
        train_filenames = readlines(data_path.parent / "kitti_splits" / self.conf['training_split'] / "train_files.txt")
        val_filenames = readlines(data_path.parent / "kitti_splits" / self.conf['training_split'] / "val_files.txt")
        img_ext = '.png' if self.conf['train_from_png'] else '.jpg'

        num_train_samples = len(train_filenames)
        self.num_total_steps = num_train_samples // self.conf['bs'] * self.conf['num_epochs'] # total number of iterations
        
        self.model_optimizer = optim.AdamW(self.parameters_to_train, self.conf['learning_rate'], weight_decay=self.conf['weight_decay']) 
        self.model_lr_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, self.conf['scheduler_step_size'], self.conf['scheduler_gamma']) 

        train_dataset = self.dataset(self.conf['data_path'], train_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=True, img_ext=img_ext) 
        self.train_loader = DataLoader(train_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)

        val_dataset = self.dataset(self.conf['data_path'], val_filenames, self.conf['im_sz'][0], self.conf['im_sz'][1], self.conf['frame_ids_training'], self.num_scales, is_train=False, img_ext=img_ext) 
        self.val_loader = DataLoader(val_dataset, self.conf['bs'], True, num_workers=self.conf['num_workers'], pin_memory=True, drop_last=True)
        self.val_iter = iter(self.val_loader)

        # Initialize fixed visualization samples if enabled
        if self.conf['use_fixed_vis_samples']:
            self.fixed_vis_samples = {
                'train': None,
                'val': None
            }
            self.initialize_fixed_vis_samples()
        else:
            self.fixed_vis_samples = None

        self.writers = {}
        for mode in ["train", "val"]:
            self.writers[mode] = SummaryWriter(os.path.join(self.log_path, mode))

        # Layer to compute the SSIM loss between a pair of images.
        if self.conf['use_ssim']:
            self.ssim = SSIM()
            self.ssim.to(self.device)

        self.backproject_depth = {}
        self.project_3d = {}
        for scale in self.conf['loss_scales']:
            h = self.conf['im_sz'][0] // (2 ** scale) 
            w = self.conf['im_sz'][1] // (2 ** scale) 

            # Layer to transform a depth image into a point cloud.
            self.backproject_depth[scale] = BackprojectDepth(self.conf['bs'], h, w)
            self.backproject_depth[scale].to(self.device)

            # Layer which projects 3D points into a camera with intrinsics K and at position T.
            self.project_3d[scale] = Project3D(self.conf['bs'], h, w)
            self.project_3d[scale].to(self.device)

        self.depth_metric_names = ["standard_metrics/abs_rel", "standard_metrics/sq_rel", "standard_metrics/rms", "standard_metrics/log_rms", "threshold_metrics/a1", "threshold_metrics/a2", "threshold_metrics/a3"]

        self.save_opts()

    def initialize_fixed_vis_samples(self):
        """
        Initialize fixed samples for visualization in TensorBoard.
        These samples will be used consistently across all logging steps.
        """
        num_samples = self.conf['num_tensorboard_samples']

        # Get fixed training samples
        train_iter = iter(self.train_loader)
        train_batch = next(train_iter)
        self.fixed_vis_samples['train'] = {}
        for key, value in train_batch.items():
            if isinstance(value, torch.Tensor):
                self.fixed_vis_samples['train'][key] = value[:num_samples].clone()
            else:
                self.fixed_vis_samples['train'][key] = value
        
        # Get fixed validation samples
        val_iter = iter(self.val_loader)
        val_batch = next(val_iter)
        self.fixed_vis_samples['val'] = {}
        for key, value in val_batch.items():
            if isinstance(value, torch.Tensor):
                self.fixed_vis_samples['val'][key] = value[:num_samples].clone()
            else:
                self.fixed_vis_samples['val'][key] = value

    def set_train(self):
        """
            Convert all models to training mode except the target encoder.
        """
        for name, m in self.models.items():
            if name == "target_encoder":
                m.eval()
            else:
                m.train()

    def set_eval(self):
        """
            Convert all models to testing/evaluation mode.
        """
        for m in self.models.values():
            m.eval()

    def train(self):
        """
            Run the entire training pipeline.
        """
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        self.save_model()
        for self.epoch in range(self.conf['num_epochs']):
            self.run_epoch()
            self.model_lr_scheduler.step()
            if (self.epoch + 1) % self.conf['save_frequency'] == 0:
                self.save_model()

    def run_epoch(self):
        """
            Run a single epoch of training and validation.
        """
        print("Training")
        self.set_train()

        metrics = {}

        for batch_idx, inputs in enumerate(self.train_loader):
            before_optimization_time = time.time()

            outputs_dict, losses = self.process_batch(inputs)

            self.model_optimizer.zero_grad()
            losses["loss"].backward()
            
            if self.conf['clip_grad_norm'] is not None: 
                torch.nn.utils.clip_grad_norm_(self.parameters_to_train, self.conf['clip_grad_norm'])
            self.model_optimizer.step()

            duration_optimization = time.time() - before_optimization_time

            current_lr = self.model_optimizer.param_groups[0]['lr']
            [self.writers[split].add_scalar("learning_rate", current_lr, self.step) for split in ("train", "val")]


            # log less frequently after the first 2000 steps to save time & disk space:
            #  - log every 10 batches if step < 2000, otherwise log every 1000 steps
            #  - 1 step = 1 iteration = 1 batch
            early_phase = batch_idx % self.conf['log_frequency'] == 0 and self.step < 2000
            late_phase = self.step % 1000 == 0

            if early_phase or late_phase:
                self.log_time(batch_idx, duration_optimization, losses["loss"].cpu().data)
                if "depth_gt" in inputs:
                    compute_depth_metrics(inputs, outputs_dict, metrics, self.depth_metric_names) # compute depth metrics for a batch
                self.log("train", inputs,  outputs_dict, losses, metrics)
                self.val()

            self.step += 1

    def process_batch(self, inputs):
        """
            Pass a minibatch through the network and generate images and losses.
        """
        for key, ipt in inputs.items():
            inputs[key] = ipt.to(self.device)
            
        disp_maps = self.models["depth_model"](inputs["color_aug", 0, 0])
           
        # Predict poses (same for both modes)
        poses = predict_poses(self.conf, self.models, inputs)
    
        # Generate warped images and compute photometric loss (same for both modes)
        outputs_dict = self.generate_images_pred(inputs, disp_maps, poses)
        losses = compute_losses(self.conf, inputs, disp_maps, outputs_dict, self.ssim)

        return outputs_dict, losses
    
    def val(self):
        """
            Validate the model on a single minibatch.
        """
        self.set_eval()
        
        metrics = {}
        
        try:
            inputs = next(self.val_iter) # for new PyTorch
        except StopIteration:
            self.val_iter = iter(self.val_loader)
            inputs = next(self.val_iter)

        with torch.no_grad():
            outputs_dict, losses = self.process_batch(inputs)
            if "depth_gt" in inputs:
                compute_depth_metrics(inputs, outputs_dict, metrics, self.depth_metric_names)
            self.log("val", inputs, outputs_dict, losses, metrics)
            del inputs, outputs_dict, losses

        self.set_train()

    def generate_images_pred(self, inputs, disp_maps, poses):
        """
            Generate the warped (reprojected) color images for a minibatch saved into the 'outputs_dict' dictionary.
            Apart from these, we also save in the 'outputs_dict' predicted disp_maps (scaled and unscaled) and also depth_maps (1/disp_maps) interpolated at the original resolution.
            'outputs_dict':
                -the predicted disp_maps unscaled interpolated at the original resolution:  outputs_dict[("disp_unscaled", 0, scale)]
                -the predicted disp_maps scaled interpolated at the original resolution:  outputs_dict[("disp_scaled", 0, scale)]
                -the predicted depth_maps interpolated at the original resolution:  outputs_dict[("depth", 0, scale)]
                -the warped (reprojected) color images for a minibatch: outputs_dict[("color", frame_id, scale)] where frame_id is in [-1, 1]
                -optional: the identity warped images (for automasking): outputs_dict[("color_identity", frame_id, scale)] where frame_id is in [-1, 1]
        """
        outputs_dict = {}
        for scale in self.conf['loss_scales']:

            disp = disp_maps[("disp", scale)]
            if self.conf['monodepthv1_multiscale']:
                source_scale = scale
            else:
                depth_model = self.models["depth_model"]
                if getattr(depth_model, "disp_upsample") == "naf":  # image-guided upsampling (NAF) of disparity to full resolution, guided by the raw full-res image; sharper depth edges than bilinear/bicubic.
                    disp = depth_model.upsample_disp(disp, inputs[("color", 0, 0)])
                else:
                    disp = F.interpolate(disp, [self.conf['im_sz'][0], self.conf['im_sz'][1]], mode=getattr(depth_model, "disp_upsample"), align_corners=True)
                source_scale = 0

            disp_scaled, depth = disp_to_depth(disp, self.min_depth, self.max_depth)

            outputs_dict[("disp_unscaled", 0, scale)] = disp
            outputs_dict[("disp_scaled", 0, scale)] = disp_scaled
            outputs_dict[("depth", 0, scale)] = depth

            for i, frame_id in enumerate(self.conf['frame_ids_training'][1:]):

                if frame_id == "s":
                    T = inputs["stereo_T"] # use stereo baseline information
                else:
                    T = poses[("cam_T_cam", 0, frame_id)] # use predicted pose between current frame (0) and neighbor frame
                
                # from the authors of https://arxiv.org/abs/1712.00175: "Learning Depth from Monocular Videos using Direct Methods"
                if self.conf['pose_model_type'] == "simple_pose_cnn" and not self.conf['use_stereo_training']:

                    axisangle = poses[("axisangle", 0, frame_id)]
                    translation = poses[("translation", 0, frame_id)]

                    inv_depth = 1 / depth
                    mean_inv_depth = inv_depth.mean(3, True).mean(2, True)

                    T = transformation_from_parameters(axisangle[:, 0], translation[:, 0] * mean_inv_depth[:, 0], frame_id < 0)

                cam_points = self.backproject_depth[source_scale](depth, inputs[("inv_K", source_scale)])
                pix_coords = self.project_3d[source_scale](cam_points, inputs[("K", source_scale)], T)

                outputs_dict[("color", frame_id, scale)] = F.grid_sample(inputs[("color", frame_id, source_scale)],
                                                                         pix_coords,
                                                                         padding_mode="border",
                                                                         align_corners=True) # align_corners=True for better quality when sampling warped color images

                if not self.conf['disable_automasking']:
                    outputs_dict[("color_identity", frame_id, scale)] = inputs[("color", frame_id, source_scale)]
 
        return outputs_dict

    def log_time(self, batch_idx, duration, loss):
        """
            Print a logging statement to the terminal.
        """
        samples_per_sec = self.conf['bs'] / duration
        time_so_far = time.time() - self.start_time
        training_time_left = (self.num_total_steps / self.step - 1.0) * time_so_far if self.step > 0 else 0
        print_string = "epoch {:>3} | batch {:>6} | examples/s: {:5.1f}" + " | loss: {:.5f} | time elapsed: {} | time left: {}"
        print(print_string.format(self.epoch, batch_idx, samples_per_sec, loss, sec_to_hm_str(time_so_far), sec_to_hm_str(training_time_left)))

    def log(self, mode, inputs, outputs_dict, losses, metrics):
        """
            Write an event to the tensorboard events file.
        """
        writer = self.writers[mode]

        # Log only the total loss
        writer.add_scalar("loss", losses["loss"], self.step)

       
        for m, v in metrics.items():
            writer.add_scalar("{}".format(m), v, self.step)

        # Decide which samples to visualize
        if self.conf['use_fixed_vis_samples']: # Use fixed samples - need to run through model again
            vis_batch_size = self.conf['num_tensorboard_samples']
            vis_inputs = self.fixed_vis_samples[mode]
            
            # Temporarily create layers with visualization batch size (we need to do this because the backprojection and projection layers depend on batch size, and we want to use the same fixed samples for visualization in both train and val modes, which may have different batch sizes than the training batch size)
            saved_backproject = self.backproject_depth
            saved_project = self.project_3d
            
            self.backproject_depth = {}
            self.project_3d = {}
            for scale in self.conf['loss_scales']:
                h = self.conf['im_sz'][0] // (2 ** scale)
                w = self.conf['im_sz'][1] // (2 ** scale)
                self.backproject_depth[scale] = BackprojectDepth(vis_batch_size, h, w).to(self.device)
                self.project_3d[scale] = Project3D(vis_batch_size, h, w).to(self.device)
            
            # Run through model 
            with torch.no_grad():
                vis_outputs_dict, _ = self.process_batch(vis_inputs)
            
            # Restore original layers 
            self.backproject_depth = saved_backproject
            self.project_3d = saved_project
        else: # Use current batch samples
            vis_inputs = inputs
            vis_outputs_dict = outputs_dict

        for j in range(self.conf['num_tensorboard_samples']):  # write a number of num_tensorboard_samples images
            if self.conf['separate_plots']: # log each input and warped image separately (instead of concatenating them into a single image)
                for frame_id in self.conf['frame_ids_training']:
                    writer.add_image("input_color_image_{}/{}".format(frame_id, j), vis_inputs[("color", frame_id, 0)][j].data, self.step)
                    if frame_id != 0:
                        writer.add_image("warped_color_image_{}/{}".format(frame_id, j), vis_outputs_dict[("color", frame_id, 0)][j].data, self.step)
            else: 
                input_imgs = []
                for frame_id in [-1, 0, 1]:
                    img = vis_inputs[("color", frame_id, 0)][j]
                    input_imgs.append(img)
                input_concat = torch.cat(input_imgs, dim=1)
                writer.add_image(f"inputs_images/{j}_frames_-1_0_1", input_concat, self.step)

                warped_imgs = []
                for frame_id in [-1, 1]:
                    warped = vis_outputs_dict[("color", frame_id, 0)][j]
                    warped_imgs.append(warped)
                warped_concat = torch.cat(warped_imgs, dim=1)
                writer.add_image(f"warped_images/{j}_frames_-1_1", warped_concat, self.step)

            if not self.conf['disable_automasking']: # auto-masking stationary pixels visualization
              writer.add_image("automask/{}".format(j), vis_outputs_dict["identity_selection/{}".format(0)][j][None, ...], self.step)

            writer.add_image("predicted_disp/{}".format(j), normalize_image(vis_outputs_dict[("disp_unscaled", 0, 0)][j]), self.step)

            # Saving color mapped depth image
            output_depth_map = vis_outputs_dict[("disp_unscaled", 0, 0)][j]
            output_depth_map_np = output_depth_map.squeeze().detach().cpu().numpy()
            vmax = np.percentile(output_depth_map_np, 95) # The 95th percentile is used here to ignore the top 5% of depth values which might be outliers and to enhance the depth visualization.
            normalizer = mpl.colors.Normalize(vmin=output_depth_map_np.min(), vmax=vmax) 
            mapper = cm.ScalarMappable(norm=normalizer, cmap='viridis')  # choices: ['viridis', 'plasma', 'inferno'].
            color_depth_map = (mapper.to_rgba(output_depth_map_np)[:, :, :3] * 255).astype(np.uint8)
            writer.add_image("predicted_depth/{}".format(j), color_depth_map.transpose(2,0,1), self.step)

    def save_opts(self):
        """
            Save configuration options to tensorboard together with models statistics.
        """
        # Save configuration to TensorBoard exactly as in the file
        yaml = YAML()
        yaml.preserve_quotes = True
        with open("src/config/config.yaml") as f:
            config = yaml.load(f)
        buffer = StringIO()
        yaml.dump(config, buffer)
        # self.writers['train'].add_text('config', buffer.getvalue())
        markdown = f"### Config\n```\n{buffer.getvalue()}\n```" 
        self.writers['train'].add_text('config', markdown)

        
        # Depth Model Statistics
        depth_total_params, depth_trainable_params = count_parameters(self.models["depth_model"])
        
        # Dynamically determine output shapes by forward pass with dummy input
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
            dummy_output = self.models["depth_model"](dummy_input)
        
        depth_stats = textwrap.dedent(f"""
            Depth Model Statistics:
            -----------------------
            Model Type: {self.conf['model_name']}
            Total Parameters: {depth_total_params:,}
            Trainable Parameters: {depth_trainable_params:,}
            Frozen Parameters: {depth_total_params - depth_trainable_params:,}

            Input Shape: [batch_size, 3, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
            """)
        
        # Determine if single-scale or multi-scale based on output type
        is_multi_scale = isinstance(dummy_output, dict)
        
        if is_multi_scale:
            depth_stats += "\nOutput Shapes (Multi-Scale):\n"
            for key in sorted(dummy_output.keys()):
                shape = dummy_output[key].shape
                depth_stats += f"  {key}: [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
        else:
            shape = dummy_output.shape
            depth_stats += f"Output Shape (Single-Scale): [batch_size, {shape[1]}, {shape[2]}, {shape[3]}]\n"
        
        self.writers['train'].add_text('depth_model_stats', depth_stats)
        
        # Pose Model Statistics
        pose_total_params, pose_trainable_params = count_parameters(self.models["pose_model"])
        
        # Dynamically determine pose model output shapes by forward pass with dummy input
        num_input_channels = 3 * self.num_pose_frames  # 3 channels per frame
        with torch.no_grad():
            dummy_pose_input = torch.randn(1, num_input_channels, self.conf['im_sz'][0], self.conf['im_sz'][1]).to(self.device)
            dummy_pose_output = self.models["pose_model"](dummy_pose_input)
        
        pose_stats = textwrap.dedent(f"""
            Pose Model Statistics:
            ----------------------
            Model Type: {self.conf['pose_model_type']}
            Total Parameters: {pose_total_params:,}
            Trainable Parameters: {pose_trainable_params:,}
            Frozen Parameters: {pose_total_params - pose_trainable_params:,}

            Input Shape: [batch_size, {num_input_channels}, {self.conf['im_sz'][0]}, {self.conf['im_sz'][1]}]
              (Number of input frames: {self.num_pose_frames})
              (Frame IDs for training: {self.conf['frame_ids_training']})
            
            Output Shapes:
            """)
        
        # Handle different pose model output formats
        if isinstance(dummy_pose_output, dict):
            for key in sorted(dummy_pose_output.keys()):
                shape = dummy_pose_output[key].shape
                shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
                pose_stats += f"  {key}: {shape_str}\n"
        elif isinstance(dummy_pose_output, (list, tuple)):
            for i, output in enumerate(dummy_pose_output):
                shape = output.shape
                output_name = "axisangle" if i == 0 else "translation"
                shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
                pose_stats += f"  {output_name}: {shape_str}\n"
        else:
            shape = dummy_pose_output.shape
            shape_str = f"[batch_size, {', '.join(str(s) for s in shape[1:])}]"
            pose_stats += f"  output: {shape_str}\n"
        
        self.writers['train'].add_text('pose_model_stats', pose_stats)

    def save_model(self):
        """
            Save model weights to disk.
        """
        save_folder = os.path.join(self.log_path, "models", "weights_epoch_{}".format(self.epoch))
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        for model_name, model in self.models.items():
            save_path = os.path.join(save_folder, "{}.pth".format(model_name))
            to_save = model.state_dict()
            torch.save(to_save, save_path)

        save_path = os.path.join(save_folder, "{}.pth".format("adam"))
        torch.save(self.model_optimizer.state_dict(), save_path)

if __name__ == "__main__":
    lt.monkey_patch()

    conf = Conf().conf  

    trainer = Trainer(conf)
    trainer.train()
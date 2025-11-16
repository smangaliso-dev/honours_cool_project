from minigrid.minigrid_env import Grid, MiniGridEnv
from minigrid.core.constants import COLOR_TO_IDX
from minigrid.core.world_object import WorldObj, Ball, Box, Key
from gymnasium.core import ObservationWrapper
from gymnasium.spaces import Dict   
import numpy as np
from random import choice
from gymnasium import spaces
from typing import Tuple, List, Any
from specs_distributions import SpecGenerator

from global_utils import DistType, FULL_DIST, BASIC_SKILLS_DIST



SHAPE_CLASSES = [Ball, Box, Key]                     # 3 shapes
COLOR_NAMES = list(COLOR_TO_IDX.keys())              # 6 colors (red, green, blue, purple, yellow, grey)
obj_types = [Ball, Box, Key]
colors = list(COLOR_TO_IDX.keys())


def target_index(shape_cls, color_name) -> int:
    """0..17 index: shape-major order."""
    s = SHAPE_CLASSES.index(shape_cls)
    c = COLOR_NAMES.index(color_name)
    return s * len(COLOR_NAMES) + c


class ObjObsWrapper(ObservationWrapper):
    
    def __init__(self, env):
        super().__init__(env)
        
        self.spec_gen = SpecGenerator()
        
        self.observation_space = Dict({
            "image": env.observation_space.spaces["image"],
            "specs": spaces.Box(low=0.0, high=1.0, shape=(9,), dtype=np.float32),
        })

    def observation(self, obs):

        target_color = self.env.unwrapped.target_color
        target_type = self.env.unwrapped.target_type
        
        # Generate the specification vector
        mission_array = self.spec_gen.generate_spec(color=target_color, obj_type=target_type)
        
        # print(f"Obs Wrapper - Target: {target_color} {target_type.__name__}, Spec: {mission_array}")
        
        wrapped_obs = {
            "image": obs["image"],
            "specs": mission_array,
        }
        return wrapped_obs

    def set_custom_spec(self, color=None, obj_type=None):
        """
        Manually set a custom specification for the environment
        Useful for curriculum learning or testing specific tasks
        """
        return self.spec_gen.generate_spec(color=color, obj_type=obj_type)


class CustomShapesEnvRaw(MiniGridEnv):

    def __init__(self, size=12, num_objects=2, max_steps=200, step_penalty = -0.01
                ,target_type: WorldObj|None=None, target_color="blue", render_mode="rgb_array", target_pos: tuple[int,int]|None=None):
        
        self.num_objects = num_objects
        self.target_type = target_type
        self.target_color = target_color
        self.step_penalty = step_penalty
        self.target_pos = target_pos

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            mission_space=spaces.Text(max_length=50, charset=spaces.text.alphanumeric),
        )

        self.action_space = spaces.Discrete(4)  # left, right, forward, pickup
        
        self.action_mapping = [
            self.actions.left,
            self.actions.right,
            self.actions.forward,
            self.actions.pickup,
        ]

    def _gen_grid(self, width, height):
        
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # place target
        if self.target_type:	target_obj = self.target_type(self.target_color)
        
        if self.target_pos:	self.put_obj(target_obj,*self.target_pos) 		# static targt position
        else:	self.target_pos = self.place_obj(target_obj)
        
        self.place_agent()

        # place distractors
        obj_types = [Ball, Box, Key]
        colors = list(COLOR_TO_IDX.keys())

        for _ in range(self.num_objects - 1):
            t = np.random.choice(obj_types)
            c = np.random.choice(colors)
            # avoid placing another target identical to goal
            if t == self.target_type and c == self.target_color:
                continue
            self.place_obj(t(c))
        
        self.target_idx = target_index(self.target_type, self.target_color)
        self.mission = f"pick up {self.target_color} {self.target_type.__name__.lower()}"

    def step(self, action):
        
        mapped = self.action_mapping[action]
        obs, reward, terminated, truncated, info = super().step(mapped)
        correct = False

        if mapped == self.actions.pickup and self.carrying is not None:
            # If pickup was attempted this step, check correctness
            if isinstance(self.carrying, self.target_type) and self.carrying.color == self.target_color:
                reward = 1.0
                terminated = True
                correct = True
            else:
                reward = -1.0
                terminated = True
        
        info["target_idx"] = self.target_idx
        info["correct_pickup"] = correct    

        # apply step penalty
        reward = round(reward + ((not correct)*self.step_penalty),2)

        return obs, reward, terminated, truncated, info
    
    
class CustomShapesEnvSimple(MiniGridEnv):
    def __init__(self, size=12, num_objects=3, max_steps:int|None=None, step_penalty=-0.01
                ,target_type=Box, target_color="blue", render_mode="rgb_array"
                ,agent_start_pos:Tuple[int,int]|None=None, agent_start_dir=0
                ,target_pos:Tuple[int,int]|None=None,dist_type=DistType.NO_DIST):
        
        self.num_objects = num_objects
        self.target_type = target_type
        self.target_color = target_color
        self.step_penalty = step_penalty
        self.target_pos = target_pos
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.dist_type          = dist_type
        
        if max_steps is None:	max_steps = num_objects * (size - 2)**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            mission_space=spaces.Text(max_length=50, charset=spaces.text.alphanumeric),
        )

        self.action_space = spaces.Discrete(4)  # left, right, forward, pickup
        
        self.action_mapping = [
            self.actions.left,
            self.actions.right,
            self.actions.forward,
            self.actions.pickup,
        ]
        
    def get_target_object(self) -> Tuple:
        
        target_obj_attributes = ()
        
        match self.dist_type:
            case DistType.FULL:
                target_obj_attributes = choice(FULL_DIST)
            case DistType.BASIC_SKILLS | DistType.BASIC_SKILLS_EXCL:
                target_obj_attributes = choice(BASIC_SKILLS_DIST)

        return target_obj_attributes
    
    def objects_to_exclude(self,target_elem:str) -> List[Any]:
        exclude_list = []
     
        if self.dist_type == DistType.BASIC_SKILLS_EXCL:
            match target_elem:
                case 'ball':
                    exclude_list = ['red', 'green']
                case 'box':
                    exclude_list = ['blue', 'purple']
                case 'key':
                    exclude_list = ['yellow', 'grey']
                case 'red' | 'green':
                    exclude_list = [Ball]
                case 'blue' | 'purple':
                    exclude_list = [Box]
                case 'yellow' | 'grey':
                    exclude_list = [Key]  
            
        return exclude_list  

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        distractor_colors  = colors.copy() 
        distractor_types   = obj_types.copy()
        elem_to_exclude = []
                
        if self.dist_type != DistType.NO_DIST:
            self.target_type, self.target_color = self.get_target_object()

        # place target
        if self.target_type and self.target_color:
            target_obj = self.target_type(self.target_color)
            distractor_colors.remove(self.target_color)
            distractor_types.remove(self.target_type)
        elif self.target_color:
            elem_to_exclude = self.objects_to_exclude(self.target_color)
            target_obj = np.random.choice([x for x in obj_types if x not in elem_to_exclude])(self.target_color)
            distractor_colors.remove(self.target_color)
            distractor_types = [t for t in distractor_types if t not in elem_to_exclude]
        elif self.target_type:
            elem_to_exclude = self.objects_to_exclude(self.target_type.__name__.lower())
            target_obj = self.target_type(np.random.choice([x for x in colors if x not in elem_to_exclude]))
            distractor_types.remove(self.target_type)
            distractor_colors = [c for c in distractor_colors if c not in elem_to_exclude]
        else:   raise ValueError("Either target_type or target_color must be specified.") 
        
        # if self.target_pos:	self.put_obj(target_obj,*self.target_pos) 		# static targt position
        # else:	
        self.target_pos = self.place_obj(target_obj)
        
        # place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # place distractors
        for _ in range(self.num_objects - 1):
            t = np.random.choice(distractor_types)
            c = np.random.choice(distractor_colors)
            # avoid placing another target identical to goal
            self.place_obj(t(c))
        
        self.mission = f"pick up a{(' ' + self.target_color) if self.target_color else ''} {self.target_type.__name__.lower() if self.target_type else 'object'}"


    def step(self, action):
        
        mapped = self.action_mapping[action]
        obs, reward, terminated, truncated, info = super().step(mapped)
        correct = False

        if mapped == self.actions.pickup and self.carrying is not None:
            # If pickup was attempted this step, check correctness
            if (self.target_type and isinstance(self.carrying, self.target_type) and (self.carrying.color == self.target_color or self.target_color is None)) \
                or (self.target_type is None and self.carrying.color == self.target_color):
                reward = 1.0
                terminated = True
                correct = True
            else:
                reward = -1.0
                terminated = True
        
        info["correct_pickup"] = correct    

        reward = float(round(reward + ((not correct) * self.step_penalty), 2))

        return obs, reward, terminated, truncated, info
    

class CustomShapesEnvSimpleEval(MiniGridEnv):
    def __init__(self, size=12, num_objects=3, max_steps:int|None=None, step_penalty=-0.01
                ,target_type=Box, target_color="blue", render_mode="rgb_array"
                ,agent_start_pos:Tuple[int,int]|None=None, agent_start_dir=0
                ,target_pos:Tuple[int,int]|None=None,dist_type=DistType.NO_DIST):
        
        self.num_objects        = num_objects
        self.target_type        = target_type
        self.target_color       = target_color
        self.step_penalty       = step_penalty
        self.target_pos         = target_pos
        self.agent_start_pos    = agent_start_pos
        self.agent_start_dir    = agent_start_dir
        self.dist_type          = dist_type
        self.init_type          = target_type
        self.init_color         = target_color

        
        if max_steps is None:	max_steps = num_objects * (size - 2)**2

        super().__init__(
            grid_size=size,
            max_steps=max_steps,
            render_mode=render_mode,
            mission_space=spaces.Text(max_length=50, charset=spaces.text.alphanumeric),
        )

        self.action_space = spaces.Discrete(3)  # left, right, forward, pickup
        
        self.action_mapping = [
            self.actions.left,
            self.actions.right,
            self.actions.forward,
            # self.actions.pickup,
        ]
        
    def get_target_object(self) -> Tuple:
        
        target_obj_attributes = ()
        
        match self.dist_type:
            case DistType.FULL:
                target_obj_attributes = choice(FULL_DIST)
            case DistType.BASIC_SKILLS | DistType.BASIC_SKILLS_EXCL:
                target_obj_attributes = choice(BASIC_SKILLS_DIST)

        return target_obj_attributes
    
    def objects_to_exclude(self,target_elem:str) -> List[Any]:
        exclude_list = []
     
        if self.dist_type == DistType.BASIC_SKILLS_EXCL:
            match target_elem:
                case 'ball':
                    exclude_list = ['red', 'green']
                case 'box':
                    exclude_list = ['blue', 'purple']
                case 'key':
                    exclude_list = ['yellow', 'grey']
                case 'red' | 'green':
                    exclude_list = [Ball]
                case 'blue' | 'purple':
                    exclude_list = [Box]
                case 'yellow' | 'grey':
                    exclude_list = [Key]  
            
        return exclude_list 

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)
        distractor_colors  = colors.copy() 
        distractor_types   = obj_types.copy()

        if self.dist_type != DistType.NO_DIST and self.init_type is None and self.init_color is None:
            self.target_type, self.target_color = self.get_target_object()

        # place target
        if self.target_type and self.target_color:
            target_obj = self.target_type(self.target_color)
            distractor_colors.remove(self.target_color)
            distractor_types.remove(self.target_type)
        elif self.target_color:
            elem_to_exclude = self.objects_to_exclude(self.target_color)
            target_obj = np.random.choice([x for x in obj_types if x not in elem_to_exclude])(self.target_color)
            distractor_colors.remove(self.target_color)
            distractor_types = [t for t in distractor_types if t not in elem_to_exclude]
        elif self.target_type:
            elem_to_exclude = self.objects_to_exclude(self.target_type.__name__.lower())
            target_obj = self.target_type(np.random.choice([x for x in colors if x not in elem_to_exclude]))
            distractor_types.remove(self.target_type)
            distractor_colors = [c for c in distractor_colors if c not in elem_to_exclude]
        else:   raise ValueError("Either target_type or target_color must be specified.")
        
        # if self.target_pos:	self.put_obj(target_obj,*self.target_pos) 		# static targt position
        # else:	
        self.target_pos = self.place_obj(target_obj)
        
        # place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.agent_pos = self.place_agent()

        # place distractors
        for _ in range(self.num_objects - 1):
            t = np.random.choice(distractor_types)
            c = np.random.choice(distractor_colors)
            # avoid placing another target identical to goal
            self.place_obj(t(c))
        
        self.mission = f"navigate to a{(' ' + self.target_color) if self.target_color else ''} {self.target_type.__name__.lower() if self.target_type else 'object'}"
    
    
    def is_target_at_position(self, pos):
        """
        Check if the object at given position matches the target specification
        """
        if pos is None: return 0
            
        cell = self.grid.get(*pos)
        if cell is None:    return 0
            
        # Check if cell contains an object (not wall, floor, etc.)
        if hasattr(cell, 'type') and cell.type in ['ball', 'box', 'key', 'wall']:  
            if (self.target_type and isinstance(cell, self.target_type) and (cell.color == self.target_color or self.target_color is None)) \
                or (self.target_type is None and cell.color == self.target_color):  return 1
        
        return -1
    
    
    def get_forward_position(self):
        """
        Get the grid position in front of the agent
        Returns: (x, y) coordinates or None if out of bounds
        """
        
        agent_x, agent_y = self.agent_pos
        agent_dir = self.agent_dir
        
        # Calculate forward position based on direction
        if agent_dir == 0:  # Right
            forward_pos = (agent_x + 1, agent_y)
        elif agent_dir == 1:  # Down
            forward_pos = (agent_x, agent_y + 1)
        elif agent_dir == 2:  # Left
            forward_pos = (agent_x - 1, agent_y)
        elif agent_dir == 3:  # Up
            forward_pos = (agent_x, agent_y - 1)
        else:
            return None
        # if (agent_dir := self.agent_dir) and agent_dir in {0,1,2,3}:    forward_pos = tuple(self.front_pos)
        # else:   return None
        
        # Check if position is within grid bounds
        if (0 < forward_pos[0] < (self.grid.width - 1) and 
            0 < forward_pos[1] < (self.grid.height - 1)):    return forward_pos
        
        return None
    

    def step(self, action):
        
        mapped = self.action_mapping[action]
        obs, reward, terminated, truncated, info = super().step(mapped)
        correct = False
        front_pos = self.get_forward_position()

        
        if self.is_target_at_position(front_pos) == 1:
            reward = 1.0
            terminated = True
            correct = True
        elif self.is_target_at_position(front_pos) == -1:
            reward = -0.04
            # terminated = True
        
        info["correct_pickup"] = correct    

        reward = float(round(reward + ((not correct) * self.step_penalty), 2))

        return obs, reward, terminated, truncated, info
    

# class MiniGridTransformerExtractor(BaseFeaturesExtractor):
#     """
#     Transformer feature extractor optimized for MiniGrid's abstract symbolic observations
#     The input image is (H, W, 3) but represents symbolic grid states, not natural images
#     """
#     def __init__(self, observation_space: Dict, features_dim: int = 256, 
#                  d_model: int = 128, nhead: int = 8, num_layers: int = 3, 
#                  dropout: float = 0.1):
        
#         super().__init__(observation_space, features_dim)
        
#         self.d_model = d_model
        
#         # MiniGrid symbolic image properties
#         img_space = observation_space.spaces["image"]
#         self.img_height = img_space.shape[0]  # Typically 7x7 or similar
#         self.img_width = img_space.shape[1]
#         self.img_channels = img_space.shape[2]  # 3 channels but symbolic
        
#         # For symbolic MiniGrid images, we treat each grid cell as a token
#         self.num_tokens = self.img_height * self.img_width
        
#         # Grid cell embedding - each cell gets embedded based on its symbolic content
#         # MiniGrid uses 3 channels with discrete values representing object types, colors, states
#         self.cell_embedding = nn.Sequential(
#             nn.Linear(self.img_channels, 32),  # Embed the 3 symbolic channels
#             nn.ReLU(),
#             nn.Linear(32, d_model),
#             nn.LayerNorm(d_model)
#         )
        
#         # Spec processing (your 9-dimensional spec vector)
#         spec_space = observation_space.spaces["specs"]
#         self.spec_dim = spec_space.shape[0]
#         self.spec_proj = nn.Sequential(
#             nn.Linear(self.spec_dim, d_model),
#             nn.LayerNorm(d_model),
#             nn.ReLU(),
#             nn.Dropout(dropout)
#         )
        
#         # CLS token for aggregation
#         self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        
#         # Positional encoding for grid positions
#         self.pos_encoding = self._create_grid_pos_encoding()
        
#         # Transformer encoder
#         encoder_layer = nn.TransformerEncoderLayer(
#             d_model=d_model,
#             nhead=nhead,
#             dim_feedforward=d_model * 2,  # Smaller for symbolic data
#             dropout=dropout,
#             activation='relu',  # ReLU works better for symbolic data
#             batch_first=True
#         )
#         self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
#         # Output projection
#         self.output_proj = nn.Sequential(
#             nn.LayerNorm(d_model),
#             nn.Linear(d_model, features_dim),
#             nn.Tanh()
#         )
        
#         # Initialize weights
#         self._initialize_weights()

#     def _create_grid_pos_encoding(self):
#         """Create 2D positional encoding for grid cells"""
#         pos_encoding = torch.zeros(self.num_tokens, self.d_model)
#         position = torch.arange(0, self.num_tokens).float().unsqueeze(1)
        
#         # Create 2D grid positions
#         div_term = torch.exp(torch.arange(0, self.d_model, 2).float() * 
#                            (-math.log(10000.0) / self.d_model))
        
#         pos_encoding[:, 0::2] = torch.sin(position * div_term)
#         pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
#         return nn.Parameter(pos_encoding.unsqueeze(0))  # (1, num_tokens, d_model)

#     def _initialize_weights(self):
#         nn.init.normal_(self.cls_token, std=0.02)
#         nn.init.xavier_uniform_(self.cell_embedding[0].weight)
#         nn.init.xavier_uniform_(self.cell_embedding[2].weight)

#     def forward(self, observations) -> torch.Tensor:
#         batch_size = observations["image"].shape[0]
        
#         # Process symbolic grid image
#         # Shape: (B, H, W, C) where C=3 contains symbolic information
#         grid_image = observations["image"].float()
        
#         # Flatten grid cells and embed each cell
#         grid_flat = grid_image.view(batch_size, -1, self.img_channels)  # (B, H*W, 3)
#         cell_embeddings = self.cell_embedding(grid_flat)  # (B, num_tokens, d_model)
        
#         # Add positional encoding
#         cell_embeddings = cell_embeddings + self.pos_encoding
        
#         # Process spec
#         spec = observations["specs"].float()
#         spec_embed = self.spec_proj(spec).unsqueeze(1)  # (B, 1, d_model)
        
#         # CLS token
#         cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # (B, 1, d_model)
        
#         # Combine: [CLS, SPEC, GRID_CELL_1, GRID_CELL_2, ...]
#         sequence = torch.cat([cls_tokens, spec_embed, cell_embeddings], dim=1)
        
#         # Apply transformer
#         encoded = self.transformer(sequence)
        
#         # Use CLS token output as features
#         cls_output = encoded[:, 0, :]
        
#         return self.output_proj(cls_output)

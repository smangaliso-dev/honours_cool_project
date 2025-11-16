#%%
import numpy as np
from minigrid.core.world_object import Ball, Box, Key


class SpecGenerator:
    
    def __init__(self):
        self.color_one_hot_dict = {
            "red": np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            "green": np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0]),
            "blue": np.array([0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
            "purple": np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]),
            "yellow": np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
            "grey": np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
            None: np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # No color specified
        }
        
        self.obj_one_hot_dict = {
            Ball: np.array([1.0, 0.0, 0.0]),
            Box: np.array([0.0, 1.0, 0.0]),
            Key: np.array([0.0, 0.0, 1.0]),
            None: np.array([0.0, 0.0, 0.0]),  # No object specified
        }
        
        self.color_from_one_hot = {tuple(v): k for k, v in self.color_one_hot_dict.items() if k is not None}
        self.obj_from_one_hot = {tuple(v): k for k, v in self.obj_one_hot_dict.items() if k is not None}
    
    def generate_spec(self, color=None, obj_type=None):
        """
        Generate a 9-dimensional specification vector
        
        Args:
            color: str or None - one of "red", "green", "blue", "purple", "yellow", "grey", or None for any color
            obj_type: class or None - Ball, Box, Key, or None for any object type
            
        Returns:
            np.array: 9-dimensional specification vector
        """
        color_vec = self.color_one_hot_dict[color]
        obj_vec = self.obj_one_hot_dict[obj_type]
        
        return np.concatenate([color_vec, obj_vec])
    
    def decode_spec(self, spec_vector):
        """
        Decode a 9-dimensional specification vector back to color and object type
        
        Args:
            spec_vector: np.array of shape (9,) - the specification vector
            
        Returns:
            tuple: (color, obj_type) where either can be None
        """
        if len(spec_vector) != 9:
            raise ValueError(f"Expected 9-dimensional vector, got {len(spec_vector)}")
            
        color_vec = spec_vector[:6]
        obj_vec = spec_vector[6:]
        
        # Decode color
        color_tuple = tuple(color_vec)
        color = self.color_from_one_hot.get(color_tuple, None)
        
        # Decode object type
        obj_tuple = tuple(obj_vec)
        obj_type = self.obj_from_one_hot.get(obj_tuple, None)
        
        return color, obj_type
    
    def get_all_possible_specs(self):
        """
        Generate all possible 9-dimensional specification combinations
        
        Returns:
            list: All possible (color, obj_type, spec_vector) combinations
        """
        colors = ["red", "green", "blue", "purple", "yellow", "grey", None]
        obj_types = [Ball, Box, Key, None]
        
        all_specs = []
        for color in colors:
            for obj_type in obj_types:
                spec_vector = self.generate_spec(color, obj_type)
                all_specs.append((color, obj_type, spec_vector))
        
        return all_specs
    
    def get_spec_by_category(self, category):
        """
        Get specifications by category type
        
        Args:
            category: str - one of "color_only", "type_only", "specific", "any"
            
        Returns:
            list: Filtered specifications
        """
        all_specs = self.get_all_possible_specs()
        
        if category == "color_only":
            return [(c, o, s) for c, o, s in all_specs if c is not None and o is None]
        elif category == "type_only":
            return [(c, o, s) for c, o, s in all_specs if c is None and o is not None]
        elif category == "specific":
            return [(c, o, s) for c, o, s in all_specs if c is not None and o is not None]
        elif category == "any":
            return [(c, o, s) for c, o, s in all_specs if c is None and o is None]
        else:
            raise ValueError("Category must be 'color_only', 'type_only', 'specific', or 'any'")

# Usage examples
def demonstrate_usage():
    
    spec_gen = SpecGenerator()
    
    # # Example 1: Specific color and object
    # print("\n1. Specific target: Red Ball")
    # spec1 = spec_gen.generate_spec(color="red", obj_type=Ball)
    # print(f"Spec vector: {spec1}")
    # decoded = spec_gen.decode_spec(spec1)
    # print(f"Decoded: {decoded}")
    
    # # Example 2: Color only (any object of that color)
    # print("\n2. Color only: Any Blue object")
    # spec2 = spec_gen.generate_spec(color="blue", obj_type=None)
    # print(f"Spec vector: {spec2}")
    # decoded = spec_gen.decode_spec(spec2)
    # print(f"Decoded: {decoded}")
    
    # Example 3: Object type only (any color of that type)
    print("\n3. Type only: Any Box (any color)")
    spec3 = spec_gen.generate_spec(color=None, obj_type=Ball)
    print(f"Spec vector: {spec3}")
    decoded = spec_gen.decode_spec(spec3)
    print(f"Decoded: {decoded}")
    
    # # Example 4: Any object (no specification)
    # print("\n4. Any object: No specification")
    # spec4 = spec_gen.generate_spec(color=None, obj_type=None)
    # print(f"Spec vector: {spec4}")
    # decoded = spec_gen.decode_spec(spec4)
    # print(f"Decoded: {decoded}")
    
    # Show all possible combinations
    # print("\n=== ALL POSSIBLE COMBINATIONS ===")
    # all_specs = spec_gen.get_all_possible_specs()
    # print(f'Total specs generated: {all_specs}')
    # print(f"Total combinations: {len(all_specs)}")
    
    # # Show by categories
    # print("\n=== BY CATEGORIES ===")
    # categories = ["color_only", "type_only", "specific", "any"]
    # for category in categories:
    #     specs = spec_gen.get_spec_by_category(category)
    #     print(f"{category}: {len(specs)} specs")
    #     for color, obj_type, spec in specs:
    #         obj_name = obj_type.__name__ if obj_type else "Any"
    #         print(f"  - {color or 'Any'} {obj_name} -> Spec: {spec}")


if __name__ == "__main__":
    demonstrate_usage()
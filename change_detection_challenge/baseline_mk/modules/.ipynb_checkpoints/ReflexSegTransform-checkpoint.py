## TO TEST

import albumentations as A
from albumentations.core.transforms_interface import DualTransform

class ReflexSegTransform(DualTransform):
    def __init__(self, always_apply = False, p = 0.5):
        """
        Initialize your custom transformation with any parameters you need.

        Parameters:
            - your_parameters_here: Add any parameters specific to your transformation.
            - always_apply: Whether to apply the transformation always (default: False).
            - p: Probability of applying the transformation (default: 1.0).
        """
        super(ReflexSegTransform, self).__init__(always_apply, p)

    def apply(self, image, **params):
        """
        Implement the transformation logic for the image.

        Parameters:
            - image: Input image.
            - **params: Additional parameters.

        Returns:
            - Transformed image.
        """
        # Implement your transformation logic here
        # You can use OpenCV, NumPy, or any other image processing libraries
        # Example: Your custom image processing code
        # transformed_image = custom_function(image, self.your_parameters_here)
        width, height = *image.shape
        transformed_image = image
        for x in range(width/2):
            for y in height:
                # SWAP
                transformed_image[x][y], transformed_image[width-1-x][y] = transformed_image[width-1-x][y], transformed_image[x][y]
        
        return transformed_image

    def apply_to_mask(self, mask, **params):
        """
        Implement the transformation logic for the mask.

        Parameters:
            - mask: Input mask.
            - **params: Additional parameters.

        Returns:
            - Transformed mask.
        """
        # Implement your transformation logic for masks here
        # Example: Your custom mask processing code
        # transformed_mask = custom_mask_function(mask, self.your_parameters_here)

        transformed_mask = mask
        transformed_mask[mask == 1] = 2
        transformed_mask[mask == 2] = 1
        
        return transformed_mask

    def get_transform_init_args_names(self):
        """
        Return the names of the parameters required to initialize the transform.

        Returns:
            - Tuple of parameter names.
        """
        return ('always_apply', 'p')
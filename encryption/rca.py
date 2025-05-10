import numpy as np

def generate_rca_rules():
    rca_rules = {}
    
    rules = [
        ((0, 0, 0, 0, 0, 0), 1), ((0, 0, 0, 0, 0, 1), 0),
        ((0, 0, 0, 0, 1, 0), 1), ((0, 0, 0, 0, 1, 1), 0),
        ((0, 0, 0, 1, 0, 0), 1), ((0, 0, 0, 1, 0, 1), 0),
        ((0, 0, 0, 1, 1, 0), 1), ((0, 0, 0, 1, 1, 1), 0),
        ((0, 0, 1, 0, 0, 0), 0), ((0, 0, 1, 0, 0, 1), 1),
        ((0, 0, 1, 0, 1, 0), 0), ((0, 0, 1, 0, 1, 1), 1),
        ((0, 0, 1, 1, 0, 0), 1), ((0, 0, 1, 1, 0, 1), 0),
        ((0, 0, 1, 1, 1, 0), 1), ((0, 0, 1, 1, 1, 1), 0),
        ((0, 1, 0, 0, 0, 0), 0), ((0, 1, 0, 0, 0, 1), 1),
        ((0, 1, 0, 0, 1, 0), 0), ((0, 1, 0, 0, 1, 1), 1),
        ((0, 1, 0, 1, 0, 0), 0), ((0, 1, 0, 1, 0, 1), 1),
        ((0, 1, 0, 1, 1, 0), 0), ((0, 1, 0, 1, 1, 1), 1),
        ((0, 1, 1, 0, 0, 0), 1), ((0, 1, 1, 0, 0, 1), 0),
        ((0, 1, 1, 0, 1, 0), 1), ((0, 1, 1, 0, 1, 1), 0),
        ((0, 1, 1, 1, 0, 0), 1), ((0, 1, 1, 1, 0, 1), 0),
        ((0, 1, 1, 1, 1, 0), 1), ((0, 1, 1, 1, 1, 1), 0),
        ((1, 0, 0, 0, 0, 0), 0), ((1, 0, 0, 0, 0, 1), 1),
        ((1, 0, 0, 0, 1, 0), 0), ((1, 0, 0, 0, 1, 1), 1),
        ((1, 0, 0, 1, 0, 0), 0), ((1, 0, 0, 1, 0, 1), 1),
        ((1, 0, 0, 1, 1, 0), 0), ((1, 0, 0, 1, 1, 1), 1),
        ((1, 0, 1, 0, 0, 0), 0), ((1, 0, 1, 0, 0, 1), 1),
        ((1, 0, 1, 0, 1, 0), 0), ((1, 0, 1, 0, 1, 1), 1),
        ((1, 0, 1, 1, 0, 0), 1), ((1, 0, 1, 1, 0, 1), 0),
        ((1, 0, 1, 1, 1, 0), 1), ((1, 0, 1, 1, 1, 1), 0),
        ((1, 1, 0, 0, 0, 0), 0), ((1, 1, 0, 0, 0, 1), 1),
        ((1, 1, 0, 0, 1, 0), 0), ((1, 1, 0, 0, 1, 1), 1),
        ((1, 1, 0, 1, 0, 0), 1), ((1, 1, 0, 1, 0, 1), 0),
        ((1, 1, 0, 1, 1, 0), 1), ((1, 1, 0, 1, 1, 1), 0),
        ((1, 1, 1, 0, 0, 0), 0), ((1, 1, 1, 0, 0, 1), 1),
        ((1, 1, 1, 0, 1, 0), 0), ((1, 1, 1, 0, 1, 1), 1),
        ((1, 1, 1, 1, 0, 0), 0), ((1, 1, 1, 1, 0, 1), 1),
        ((1, 1, 1, 1, 1, 0), 0), ((1, 1, 1, 1, 1, 1), 1), 
    ]

    for condition, next_state in rules:
        rca_rules[condition] = int(next_state)
        
    return rca_rules

def apply_rca(image, rca_mask, iterations=1, reverse=False):
    """Apply RCA diffusion on 4 MSBs of each pixel in a 2D manner using wrap padding."""
    height, width = image.shape
    rca_rules = generate_rca_rules() 

    binary_image = np.unpackbits(image.reshape(-1, 1), axis=1)
    msb_image = binary_image[:, :4]  # Extract 4 MSBs
    lsb_image = binary_image[:, 4:]  # Keep LSBs unchanged

    # Reshape MSB array to 2D format
    msb_image = msb_image.reshape(height, width, 4)

    for _ in range(iterations):
        new_msb_image = np.copy(msb_image)

        for y in range(height):
            for x in range(width):
                # Wrap around at the edges (Toroidal Boundary)
                left   = msb_image[y, (x - 1) % width, 3]  # Left MSB
                right  = msb_image[y, (x + 1) % width, 3]  # Right MSB
                top    = msb_image[(y - 1) % height, x, 3] # Top MSB
                bottom = msb_image[(y + 1) % height, x, 3] # Bottom MSB
                center = msb_image[y, x, :]  # Full 4-bit MSB array

                # Use chaotic mask only in the first iteration
                mask_bit = rca_mask[y, x] if _ == 0 else new_msb_image[y, x, 3]

                key = (left, top, center[3], bottom, right, mask_bit)

                if reverse:
                    # Reverse lookup: find the original state before encryption
                    original_state = next((v for k, v in rca_rules.items() if k == key), center[3])
                    new_msb_image[y, x, 3] = original_state  # Restore original MSB
                else:
                    # Apply RCA rule for forward encryption
                    new_msb_image[y, x, 3] = rca_rules.get(key, center[3])  # Modify 1st MSB

        msb_image = new_msb_image  # Update for next iteration

    # Flatten and recombine MSBs and LSBs
    final_binary_image = np.concatenate((msb_image.reshape(-1, 4), lsb_image), axis=1)
    return np.packbits(final_binary_image, axis=1).reshape(image.shape)

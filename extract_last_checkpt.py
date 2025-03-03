import torch

def extract_last_checkpoint(input_pth, output_pth):
    # Load the checkpoint list
    checkpoint_lst = torch.load(input_pth, map_location="cpu")

    # Ensure the loaded object is a list
    if not isinstance(checkpoint_lst, list):
        raise TypeError("Expected a list of checkpoints, but got a different format.")

    # Extract the last checkpoint dictionary
    last_checkpoint = checkpoint_lst[-1]

    # Ensure it's a dictionary
    if not isinstance(last_checkpoint, dict):
        raise TypeError("Last checkpoint is not a dictionary.")

    # Convert model weights to float16
    model_state_dict = last_checkpoint["model_state_dict"]
    model_state_dict = {k: v.half() for k, v in model_state_dict.items()}

    # Save the reduced-precision checkpoint
    torch.save(model_state_dict, output_pth)
    print(f"Last checkpoint saved with float16 precision to {output_pth}")

# Example usage
input_checkpoint = r"C:\Users\samee\OneDrive\Documents\VNIT !!!\IvLabs\IvLabs Codes\Facial_LandmarksDetection\FLD_params2.pth"  # Replace with your actual checkpoint file
output_checkpoint = "latest_params.pth"  # The new output file
extract_last_checkpoint(input_checkpoint, output_checkpoint)

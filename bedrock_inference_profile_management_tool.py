import os
import boto3
import csv
import argparse
from datetime import datetime
from bedrock_tagger import BedrockTagger

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bedrock Inference Profile Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new inference profile (interactive mode)
  python3 create_inference_profile_interactive.py
  
  # List and manage existing inference profiles
  python3 create_inference_profile_interactive.py -l
  
Operations:
  - Create new inference profiles with tags
  - List existing application inference profiles
  - Delete existing profiles
  - Support both Foundation Models and Inference Profiles(including Cross-region Inference Profiles)
  - Export profile information to CSV
        """)
    
    parser.add_argument(
        '-l', '--list',
        action='store_true',
        help='List all Application inference profiles and provide option to delete'
    )
    
    return parser.parse_args()

def initBoto3Session() -> boto3.Session:
    profiles = boto3.Session().available_profiles
    if profiles:
        print("\n=== Choose Credential Profile ===")
        for idx, profile in enumerate(profiles):
            print(f"{idx}. {profile}")
        profile_index = int(get_user_input("Please select credential profile to use","0"))
        profile_name = profiles[profile_index]
        _session = boto3.Session(profile_name=profile_name)
        # Use profile credentials
        credentials = _session.get_credentials()
        # Set env
        os.environ['AWS_ACCESS_KEY_ID'] = credentials.access_key
        os.environ['AWS_SECRET_ACCESS_KEY'] = credentials.secret_key

    else:
        print("\n=== Input AWS Credential Information ===")
        # Set env
        os.environ['AWS_ACCESS_KEY_ID'] = get_user_input("Please input AWS Access Key ID")
        os.environ['AWS_SECRET_ACCESS_KEY'] = get_user_input("Please input AWS Secret Access Key")

    region = get_user_input("Please input Region", "us-west-2")
    os.environ['AWS_REGION'] = region

    session = boto3.Session(aws_access_key_id=os.environ['AWS_ACCESS_KEY_ID'], 
                            aws_secret_access_key=os.environ['AWS_SECRET_ACCESS_KEY'], 
                            region_name=region)
    return session

def get_user_input(prompt: str, default: str = None) -> str:
    """Get user input and support the default value."""
    if default:
        user_input = input(f"{prompt} [{default}]: ").strip()
        return user_input if user_input else default
    return input(f"{prompt}: ").strip()

def get_tags_input() -> list:
    """Interactive access to tag information"""
    tags = []
    while True:
        if tags and get_user_input("Continue adding tags?(y/n)", "n").lower() != 'y':
            break
        key = get_user_input("Please input the tag key", "map-migrated")
        value = get_user_input("Please input the tag value")
        tags.append({'key': key, 'value': value})
    return tags

def list_available_models(session, keyword: str = None) -> list:
    """
    List available Bedrock models that support ON_DEMAND inference, optionally filtered by keyword.
    
    Args:
        session: boto3.Session object
        keyword: Optional string to filter model names
        
    Returns:
        list of dictionaries containing model information
    """
    try:
        bedrock = session.client('bedrock')
        response = bedrock.list_foundation_models()
        models = []
        
        for model in response['modelSummaries']:
            model_id = model['modelId']
            inference_types = model.get('inferenceTypesSupported', [])
            
            # Only include models that support ON_DEMAND inference
            if 'ON_DEMAND' in inference_types:
                if keyword is None or keyword.lower() in model_id.lower():
                    models.append({
                        'modelId': model_id,
                        'providerName': model.get('providerName', 'N/A'),
                        'modelName': model.get('modelName', 'N/A')
                    })
        
        return models
    except Exception as e:
        print(f"Error listing models: {str(e)}")
        return []

def display_models(models: list):
    """Display models in a formatted way"""
    if not models:
        print("No models found.")
        return
        
    print("\n=== Available Models ===")
    print(f"Found {len(models)} models:")
    for idx, model in enumerate(models):
        print(f"\n{idx}. Model ID: {model['modelId']}")
        print(f"   Provider: {model['providerName']}")
        print(f"   Name: {model['modelName']}")

def get_valid_models(session: boto3.Session) -> list:
    """
    Keep asking for keyword until we get some models to display
    
    Args:
        session: boto3.Session object
        
    Returns:
        list of models
    """
    while True:
        keyword = get_user_input("Enter keyword to filter models", "")
        if not keyword:
            print("Please enter a valid keyword.")
            continue
        models = list_available_models(session, keyword)
        
        if models:
            display_models(models)
            return models
        else:
            print("\nNo models found with the given keyword. Please try again.")        

def list_inference_profiles(session, region: str = None, type: str = 'SYSTEM_DEFINED') -> list:
    """
    List all inference profiles across regions.
    
    Args:
        session: boto3.Session object
        region: Optional string to filter by specific region
        
    Returns:
        list of dictionaries containing profile information
    """
    profiles = []

    try:
        bedrock = session.client('bedrock', region_name=region)
        paginator = bedrock.get_paginator('list_inference_profiles')

        for page in paginator.paginate(typeEquals=type):
            for profile in page.get('inferenceProfileSummaries', []):
                # Get tags for each profile
                try:
                    tags_response = bedrock.list_tags_for_resource(
                        resourceARN=profile.get('inferenceProfileArn')
                    )
                    tags = tags_response.get('tags', [])
                except Exception as e:
                    tags = []
                
                profiles.append({
                    'region': region,
                    'name': profile.get('inferenceProfileName'),
                    'modelArn': profile.get('models')[0].get('modelArn'),
                    'inferenceProfileArn': profile.get('inferenceProfileArn'),
                    'modelId': profile.get('inferenceProfileId'),
                    'status': profile.get('status'),
                    'tags': tags
                })

    except Exception as e:
        print(f"Error listing profiles in region {region}: {str(e)}")
    return profiles

def display_inference_profiles(profiles: list):
    """Display inference profiles in a formatted way"""
    if not profiles:
        print("No inference profiles found.")
        return
        
    print("\n=== Available Inference Profiles ===")
    print(f"Found {len(profiles)} profiles:")
    for idx, profile in enumerate(profiles):
        print(f"\n{idx}. Profile Name: {profile['name']}")
        print(f"   Region: {profile['region']}")
        print(f"   Model ID: {profile['modelId']}")
        print(f"   Model ARN: {profile['modelArn']}")
        print(f"   Status: {profile['status']}")
        print(f"   ARN: {profile['inferenceProfileArn']}")
        if profile['tags']:
            print("   Tags:")
            for tag in profile['tags']:
                print(f"     - {tag['key']}: {tag['value']}")

def get_inference_profiles(session: boto3.Session, region: str = None) -> list:
    """
    Args:
        session: boto3.Session object
        
    Returns:
        list of inference profiles
    """
    while True:
        profiles = list_inference_profiles(session, region)
        
        if profiles:
            display_inference_profiles(profiles)
            return profiles
        else:
            print("\nNo profiles found, Please try again.")

def save_to_csv(profiles: list, tags: list, filename: str = None):
    """
    Save inference profiles and their tags to a CSV file.
    
    Args:
        profiles: list of inference profile dictionaries
        tags: list of tag dictionaries
        filename: Optional specific filename to use, if None will generate with timestamp
    """
    try:
        # Use provided filename or create one with timestamp
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"inference_profiles_{timestamp}.csv"
        
        # Format tags as a single string
        tag_str = '; '.join([f"{tag['key']}={tag['value']}" for tag in tags])
        
        # Check if file exists to determine if we need to write header
        file_exists = os.path.exists(filename)
        
        # Write to CSV in append mode
        with open(filename, 'a', newline='') as csvfile:
            fieldnames = ['Profile Name', 'Profile ARN', 'Tags']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header only if file is new
            if not file_exists:
                writer.writeheader()
            
            # Write new profiles
            for profile in profiles:
                writer.writerow({
                    'Profile Name': profile['name'],
                    'Profile ARN': profile['inferenceProfileArn'],
                    'Tags': tag_str
                })
        
        print(f"\n✅ Results saved to {filename}")
        
    except Exception as e:
        print(f"\n❌ Error saving to CSV: {str(e)}")

def delete_inference_profile(session: boto3.Session, profile_arn: str) -> bool:
    """
    Delete an inference profile by ARN
    
    Args:
        session: boto3.Session object
        profile_arn: ARN of the profile to delete
        
    Returns:
        bool: True if deletion was successful, False otherwise
    """
    try:
        bedrock = session.client('bedrock')
        bedrock.delete_inference_profile(inferenceProfileIdentifier=profile_arn)
        print(f"\n✅ Successfully deleted inference profile: {profile_arn}")
        return True
    except Exception as e:
        print(f"\n❌ Error deleting profile: {str(e)}")
        return False

def interactive_create_inference_profile():
    # 1.List profiles and select profile or input AWS Credential Information
    session = initBoto3Session()

    # 2. Set tag information
    print("\n=== Tag Configuration ===")
    tags = get_tags_input()

    # 3. Create a session-specific filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"inference_profiles_{timestamp}.csv"

    # 4. Initialize BedrockTagger 
    bedrock_manager = BedrockTagger()

    # 5. Create Inference Profile
    while True:
        try:
            print("\n=== Create Application Inference Profile ===")
            
            profile_name = get_user_input("Please input Inference Profile Name")
            type = get_user_input("Please select Model Type: Foundation Model<1> or Inference Profile<2>","1")
            model_arn =""
            if type == "1":
                # Add model listing functionality with retry logic
                print("\n=== List Available Models ===")
                models = get_valid_models(session)
                selected_model = None
                while True:
                    try:
                        model_index = int(get_user_input("Please select model index"))
                        if 0 <= model_index < len(models):
                            selected_model = models[model_index]
                            break
                        else:
                            print(f"Please enter a valid index between 0 and {len(models)-1}")
                    except ValueError:
                        print("Please enter a valid number")
                
                model_arn = f"arn:aws:bedrock:{os.environ['AWS_REGION']}::foundation-model/{selected_model['modelId']}"
                print(f"Selected model ARN: {model_arn}")

            else:
                print("\n=== List Inference Profiles ===")
                profiles = get_inference_profiles(session, os.environ['AWS_REGION'])
                selected_profile = None
                while True:
                    try:
                        profile_index = int(get_user_input("Please select profile index"))
                        if 0 <= profile_index < len(profiles):
                            selected_profile = profiles[profile_index]
                            break
                        else:
                            print(f"Please enter a valid index between 0 and {len(profiles)-1}")
                    except ValueError:
                        print("Please enter a valid number")
                
                model_arn = selected_profile['inferenceProfileArn']
                print(f"Selected profile ARN: {model_arn}")

            # Start to Create Inference Profile
            print("\nCreating Inference Profile...")
            response = bedrock_manager.create_inference_profile(profile_name, model_arn, tags)
            print(f"\n✅ Your Application Inference Profile Creation Succeeded, ARN: {response['inferenceProfileArn']}")

            # Save each successful creation immediately
            new_profile = {
                'name': profile_name,
                'inferenceProfileArn': response['inferenceProfileArn']
            }
            save_to_csv([new_profile], tags, filename=session_filename)

            # Ask if user want to continue creating.
            if get_user_input("\nContinue to create another Inference Profile?(y/n)", "n").lower() != 'y':
                print("\n Thanks for using!")
                break

        except Exception as e:
            print(f"\n❌ ERROR: {str(e)}")
            if get_user_input("\nRetry?(y/n)", "y").lower() != 'y':
                print("\n Thanks for using!")
                break

def interactive_list_inference_profile():
    session = initBoto3Session()
        
    # List application inference profiles
    print("\nListing Application inference profiles...")
    profiles = list_inference_profiles(session, os.environ['AWS_REGION'], type='APPLICATION')
    display_inference_profiles(profiles)

    # Ask if user wants to delete any profiles
    if profiles and get_user_input("\nWould you like to delete any profile? (y/n)", "n").lower() == 'y':
        while True:
            try:
                profile_index = int(get_user_input("\nPlease select profile index to delete"))
                if 0 <= profile_index < len(profiles):
                    profile = profiles[profile_index]
                    if get_user_input(f"\nConfirm deletion of profile '{profile['modelId']}'? (y/n)", "n").lower() == 'y':
                        delete_inference_profile(session, profile['modelId'])
                else:
                    print(f"Please enter a valid index between 0 and {len(profiles)-1}")
                
                if get_user_input("\nDelete another profile? (y/n)", "n").lower() != 'y':
                    print("\n Thanks for using!")
                    break
            except ValueError:
                print("Please enter a valid number")

if __name__ == "__main__":
    """Main function to handle different commands"""
    args = parse_arguments()

    if args.list:
        interactive_list_inference_profile()
    else:
        interactive_create_inference_profile()

import argparse
import boto3
import csv
import os
import yaml
from bedrock_tagger import BedrockTagger
from datetime import datetime
from getpass import getpass

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description='Bedrock Inference Profile Management Tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new inference profile (interactive mode)
  python3 bedrock_inference_profile_management_tool.py
  
  # List and manage existing inference profiles
  python3 bedrock_inference_profile_management_tool.py -l

  # Batch create inference profiles from a yaml file
  python3 bedrock_inference_profile_management_tool.py -f ./bedrock-profiles.yaml
  
  # Batch tag existing inference profiles from a yaml file
  python3 bedrock_inference_profile_management_tool.py -t ./bedrock-profiles.yaml
  
Operations:
  - Create new inference profiles with tags
  - Batch create inference profiles with tags from a yaml file
  - Batch tag existing inference profiles from a yaml file
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
    parser.add_argument(
        '-f', '--file',
        type=str,
        help='Path to a yaml file for batch creation'
    )
    parser.add_argument(
        '-t', '--tag',
        type=str,
        help='Path to a yaml file for batch tagging existing profiles'
    )
    
    return parser.parse_args()

def initBoto3Session() -> boto3.Session:
    session = boto3.Session()
    credentials = session.get_credentials()
    services = session.get_available_services()
    profiles = session.available_profiles

    # Get AWS Credential
    if credentials and 'bedrock' in services:
        # Use AWS Credential from the Profile
        if profiles:
            print("\n=== Choose AWS Credential Profile ===")
            for idx, profile in enumerate(profiles):
                print(f"{idx}. {profile}")
            profile_index = int(input("\nSelect profile [0]: ").strip() or "0")
            profile_name = profiles[profile_index]
            session = boto3.Session(profile_name=profile_name)
        # Use AWS Credential from the Role
        else:
            print("\n=== Will use AWS Credential from the Role ===")

    # Use AWS Credential from the Input AK/SK
    else:
        print("\n=== Input AWS Credential Information ===")
        # Set env
        ak = get_user_input("Enter AWS Access Key ID (hidden)", is_secret=True)
        sk = get_user_input("Enter AWS Secret Access Key (hidden)", is_secret=True)
        session = boto3.Session(aws_access_key_id=ak, aws_secret_access_key=sk)

    return session

def get_user_input(prompt: str, default: str = None, is_secret: bool = False) -> str:
    """Get user input and support the default value."""
    if is_secret:
        return getpass(f"{prompt}: ").strip()
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
        key = get_user_input("Enter the tag key", "map-migrated")
        value = get_user_input("Enter the tag value")
        tags.append({'key': key, 'value': value})
    return tags

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

def get_valid_models(bedrock_tagger: BedrockTagger) -> list:
    """
    Keep asking for keyword until we get some models to display
    
    Args:
        bedrock_tagger: BedrockTaggers object
        
    Returns:
        list of models
    """
    while True:
        keyword = get_user_input("Enter keyword to filter models", "")
        if not keyword:
            print("Please enter a valid keyword.")
            continue
        models = bedrock_tagger.list_available_models(keyword)
        
        if models:
            display_models(models)
            return models
        else:
            print("\nNo models found with the given keyword. Please try again.")

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
        if profile['modelArn']:
            print("   Model ARNs:")
            for model in profile['modelArn']:
                print(f"     - {model}")
        print(f"   Status: {profile['status']}")
        print(f"   ARN: {profile['inferenceProfileArn']}")
        if profile['tags']:
            print("   Tags:")
            for tag in profile['tags']:
                print(f"     - {tag['key']}: {tag['value']}")

def get_inference_profiles(bedrock_tagger: BedrockTagger) -> list:
    """
    Args:
        bedrock_tagger: BedrockTaggers object
        
    Returns:
        list of inference profiles
    """
    while True:
        profiles = bedrock_tagger.list_inference_profiles(type='SYSTEM_DEFINED')
        
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

def interactive_create_inference_profile():
    # 1.List profiles and select profile or input AWS Credential Information
    session = initBoto3Session()
    region = get_user_input("Enter Region", "ap-northeast-1")

    # 2. Set tag information
    print("\n=== Tag Configuration ===")
    tags = get_tags_input()

    # 3. Create a session-specific filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"inference_profiles_{timestamp}.csv"

    # 4. Initialize BedrockTagger 
    bedrock_tagger = BedrockTagger(session, region)

    # 5. Create Inference Profile
    while True:
        try:
            print("\n=== Create Application Inference Profile ===")
            
            profile_name = get_user_input("Enter Inference Profile Name")
            type = get_user_input("Select Model Type: Foundation Model<1> or Inference Profile<2>", "1")
            model_arn =""
            if type == "1":
                # Add model listing functionality with retry logic
                print("\n=== List Available Models ===")
                models = get_valid_models(bedrock_tagger)
                selected_model = None
                while True:
                    try:
                        model_index = int(get_user_input("Select model index"))
                        if 0 <= model_index < len(models):
                            selected_model = models[model_index]
                            break
                        else:
                            print(f"Please enter a valid index between 0 and {len(models)-1}")
                    except ValueError:
                        print("Please enter a valid number")
                
                model_arn = f"arn:aws:bedrock:{region}::foundation-model/{selected_model['modelId']}"
                print(f"Selected model ARN: {model_arn}")

            else:
                print("\n=== List Inference Profiles ===")
                profiles = get_inference_profiles(bedrock_tagger)
                selected_profile = None
                while True:
                    try:
                        profile_index = int(get_user_input("Select profile index"))
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
            response = bedrock_tagger.create_inference_profile(profile_name, model_arn, tags)
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
    region = get_user_input("Enter Region", "ap-northeast-1")
    bedrock_tagger = BedrockTagger(session, region)

    # List application inference profiles
    print("\nListing Application inference profiles...")
    profiles = bedrock_tagger.list_inference_profiles(type='APPLICATION')
    display_inference_profiles(profiles)

    # Ask if user wants to delete any profiles
    if profiles and get_user_input("\nWould you like to delete any profile? (y/n)", "n").lower() == 'y':
        while True:
            try:
                profile_index = int(get_user_input("\nSelect profile index to delete"))
                if 0 <= profile_index < len(profiles):
                    profile = profiles[profile_index]
                    if get_user_input(f"\nConfirm deletion of profile '{profile['modelId']}'? (y/n)", "n").lower() == 'y':
                        bedrock_tagger.delete_inference_profile(profile['modelId'])
                else:
                    print(f"Please enter a valid index between 0 and {len(profiles)-1}")
                
                if get_user_input("\nDelete another profile? (y/n)", "n").lower() != 'y':
                    print("\n Thanks for using!")
                    break
            except ValueError:
                print("Please enter a valid number")

def batch_create_inference_profiles(config_file):
    # Determine file type (YAML or JSON) and load accordingly
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"❌ Unsupported file format: {config_file}")
        return
    
    # Initialize session (may need to modify to support non-interactive credential selection)
    session = initBoto3Session()
    region = config.get('region')
    if not region:
        region = get_user_input("Enter Region", "ap-northeast-1")

    tags = config.get('tags')

    # Initialize BedrockTagger
    bedrock_tagger = BedrockTagger(session, region)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"inference_profiles_{timestamp}.csv"
    
    # Process each profile
    for profile_config in config.get('bedrock-profiles', []):
        try:
            profile_name = profile_config.get('name')
            model_type = profile_config.get('model_type')
            model_id = profile_config.get('model_id')

            print(f"\nThe processing model is: {model_id}...")
            # Construct model ARN based on type
            model_arn = ""
            if model_type == "foundation":
                model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
            elif model_type == "inference":
                # Assume model_id is already a full ARN for inference profiles
                model_arn = model_id
            
            print(f"Creating Inference Profile with model ARN: {model_arn}")
            response = bedrock_tagger.create_inference_profile(profile_name, model_arn, tags)
            print(f"✅ Inference Profile created: {response['inferenceProfileArn']}")

            # Save to CSV
            new_profile = {
                'name': profile_name,
                'inferenceProfileArn': response['inferenceProfileArn']
            }
            save_to_csv([new_profile], tags, filename=session_filename)
            
        except Exception as e:
            print(f"❌ Error creating profile {profile_name}: {str(e)}")
            continue

def batch_tag_inference_profiles(config_file):
    """
    Batch tag existing inference profiles from a YAML configuration file
    
    Args:
        config_file: Path to the YAML configuration file
    """
    # Load configuration file
    if config_file.endswith('.yaml') or config_file.endswith('.yml'):
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"❌ Unsupported file format: {config_file}")
        return
    
    # Initialize session
    session = initBoto3Session()
    region = config.get('region')
    if not region:
        region = get_user_input("Enter Region", "ap-northeast-1")

    tags = config.get('tags')
    if not tags:
        print("❌ No tags found in configuration file")
        return

    # Initialize BedrockTagger
    bedrock_tagger = BedrockTagger(session, region)

    # Create CSV file for results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_filename = f"tagged_profiles_{timestamp}.csv"
    
    # Check if we have existing profiles to tag or profiles to create and tag
    existing_profiles_to_tag = config.get('existing-profiles-to-tag', [])
    profiles_to_create = config.get('bedrock-profiles', [])
    
    tagged_profiles = []
    
    # Tag existing profiles
    if existing_profiles_to_tag:
        print(f"\n=== Tagging {len(existing_profiles_to_tag)} existing profiles ===")
        for profile_config in existing_profiles_to_tag:
            try:
                profile_name = profile_config.get('name')
                profile_arn = profile_config.get('arn')
                
                if not profile_name and not profile_arn:
                    print("❌ Profile must have either 'name' or 'arn' specified")
                    continue
                
                # Find profile by name if ARN not provided
                if not profile_arn:
                    print(f"🔍 Finding profile by name: {profile_name}")
                    profile = bedrock_tagger.find_inference_profile_by_name(profile_name)
                    if not profile:
                        print(f"❌ Profile not found: {profile_name}")
                        continue
                    profile_arn = profile['inferenceProfileArn']
                    profile_name = profile['name']
                else:
                    # Get profile details by ARN
                    profile_details = bedrock_tagger.get_inference_profile_by_arn(profile_arn)
                    if not profile_details:
                        print(f"❌ Profile not found: {profile_arn}")
                        continue
                    profile_name = profile_details.get('inferenceProfileName', 'Unknown')

                print(f"🏷️  Tagging profile: {profile_name} ({profile_arn})")
                
                # Tag the profile
                success = bedrock_tagger.tag_inference_profile(profile_arn, tags)
                if success:
                    print(f"✅ Successfully tagged profile: {profile_name}")
                    tagged_profiles.append({
                        'name': profile_name,
                        'inferenceProfileArn': profile_arn
                    })
                else:
                    print(f"❌ Failed to tag profile: {profile_name}")
                    
            except Exception as e:
                print(f"❌ Error processing profile {profile_config}: {str(e)}")
                continue
    
    # Create and tag new profiles (existing functionality)
    if profiles_to_create:
        print(f"\n=== Creating and tagging {len(profiles_to_create)} new profiles ===")
        for profile_config in profiles_to_create:
            try:
                profile_name = profile_config.get('name')
                model_type = profile_config.get('model_type')
                model_id = profile_config.get('model_id')

                print(f"🔨 Creating profile: {profile_name} with model: {model_id}")
                
                # Construct model ARN based on type
                model_arn = ""
                if model_type == "foundation":
                    model_arn = f"arn:aws:bedrock:{region}::foundation-model/{model_id}"
                elif model_type == "inference":
                    model_arn = model_id
                
                response = bedrock_tagger.create_inference_profile(profile_name, model_arn, tags)
                print(f"✅ Inference Profile created and tagged: {response['inferenceProfileArn']}")

                tagged_profiles.append({
                    'name': profile_name,
                    'inferenceProfileArn': response['inferenceProfileArn']
                })
                
            except Exception as e:
                print(f"❌ Error creating profile {profile_name}: {str(e)}")
                continue
    
    # Save results to CSV
    if tagged_profiles:
        save_to_csv(tagged_profiles, tags, filename=session_filename)
        print(f"\n📊 Tagged {len(tagged_profiles)} profiles successfully")
    else:
        print("\n⚠️  No profiles were tagged")

if __name__ == "__main__":
    """Main function to handle different commands"""
    args = parse_arguments()

    if args.file:
        batch_create_inference_profiles(args.file)
    elif args.tag:
        batch_tag_inference_profiles(args.tag)
    elif args.list:
        interactive_list_inference_profile()
    else:
        interactive_create_inference_profile()
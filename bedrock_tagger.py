import os

class BedrockTagger:
    def __init__(self, session=None, region_name=None):
        self.region_name = region_name
        if not session:
            raise ValueError("Session must be provided when initializing BedrockTagger")

        self.bedrock_client = session.client("bedrock", region_name=self.region_name)

    def create_inference_profile(self, profile_name, model_arn, tags):    
        """Create Inference Profile using base model ARN"""

        #check exists before create
        profile_response = self.get_inference_profile_by_name(profile_name)
        if profile_response:
            #print("Inference profile already exists")
            raise Exception("Inference profile already exists")

        response = self.bedrock_client.create_inference_profile(
            inferenceProfileName=profile_name,
            modelSource={'copyFrom': model_arn},
            tags=tags
        )
        #print("CreateInferenceProfile Response:", response['ResponseMetadata']['HTTPStatusCode']),
        #print(f"{response}\n")
        return response

    def get_inference_profile_by_name(self, profile_name):
        response = self.bedrock_client.list_inference_profiles(
            maxResults = 100,
            typeEquals = 'APPLICATION'
        )
        profile_response=''
        inferenceProfileSummaries = response['inferenceProfileSummaries']
        for inferenceProfileSummary in inferenceProfileSummaries:
            if inferenceProfileSummary['inferenceProfileName'] == profile_name:
                profile_response = self.bedrock_client.get_inference_profile(
                    inferenceProfileIdentifier=inferenceProfileSummary['inferenceProfileArn']
                )
                break
        return profile_response

    def delete_inference_profile(self, profile_arn: str) -> bool:
        """
        Delete an inference profile by ARN
        
        Args:
            session: boto3.Session object
            profile_arn: ARN of the profile to delete
            
        Returns:
            bool: True if deletion was successful, False otherwise
        """
        try:
            self.bedrock_client.delete_inference_profile(inferenceProfileIdentifier=profile_arn)
            print(f"\n✅ Successfully deleted inference profile: {profile_arn}")
            return True
        except Exception as e:
            print(f"\n❌ Error deleting profile: {str(e)}")
            return False

    def list_available_models(self, keyword: str = None) -> list:
        """
        List available Bedrock models that support ON_DEMAND inference, optionally filtered by keyword.
        
        Args:
            session: boto3.Session object
            keyword: Optional string to filter model names
            
        Returns:
            list of dictionaries containing model information
        """
        try:
            response = self.bedrock_client.list_foundation_models()
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

    def list_inference_profiles(self, type: str = None) -> list:
        """
        List all application inference profiles in current region.
            
        Returns:
            list of dictionaries containing profile information
        """
        profiles = []

        try:
            paginator = self.bedrock_client.get_paginator('list_inference_profiles')

            for page in paginator.paginate(typeEquals=type):
                for profile in page.get('inferenceProfileSummaries', []):
                    # Get tags for each profile
                    try:
                        tags_response = self.bedrock_client.list_tags_for_resource(
                            resourceARN=profile.get('inferenceProfileArn')
                        )
                        tags = tags_response.get('tags', [])
                    except Exception as e:
                        tags = []

                    profiles.append({
                        'region': self.region_name,
                        'name': profile.get('inferenceProfileName'),
                        'modelArn': [model.get('modelArn') for model in profile.get('models', [])],
                        'inferenceProfileArn': profile.get('inferenceProfileArn'),
                        'modelId': profile.get('inferenceProfileId'),
                        'status': profile.get('status'),
                        'tags': tags
                    })

        except Exception as e:
            print(f"Error listing profiles in region {self.region_name}: {str(e)}")
        return profiles

    def tag_inference_profile(self, profile_arn: str, tags: list) -> bool:
        """
        Add tags to an existing inference profile
        
        Args:
            profile_arn: ARN of the inference profile to tag
            tags: List of tag dictionaries with 'key' and 'value'
            
        Returns:
            bool: True if tagging was successful, False otherwise
        """
        try:
            # Convert tags to the format expected by AWS API
            aws_tags = [{'key': tag['key'], 'value': tag['value']} for tag in tags]
            
            self.bedrock_client.tag_resource(
                resourceARN=profile_arn,
                tags=aws_tags
            )
            return True
        except Exception as e:
            print(f"❌ Error tagging profile {profile_arn}: {str(e)}")
            return False

    def get_inference_profile_by_arn(self, profile_arn: str):
        """
        Get inference profile details by ARN
        
        Args:
            profile_arn: ARN of the inference profile
            
        Returns:
            dict: Profile information or None if not found
        """
        try:
            response = self.bedrock_client.get_inference_profile(
                inferenceProfileIdentifier=profile_arn
            )
            return response
        except Exception as e:
            print(f"❌ Error getting profile {profile_arn}: {str(e)}")
            return None

    def find_inference_profile_by_name(self, profile_name: str, profile_type: str = 'APPLICATION'):
        """
        Find inference profile by name
        
        Args:
            profile_name: Name of the inference profile to find
            profile_type: Type of profile ('APPLICATION' or 'SYSTEM_DEFINED')
            
        Returns:
            dict: Profile information or None if not found
        """
        try:
            profiles = self.list_inference_profiles(type=profile_type)
            for profile in profiles:
                if profile['name'] == profile_name:
                    return profile
            return None
        except Exception as e:
            print(f"❌ Error finding profile {profile_name}: {str(e)}")
            return None       
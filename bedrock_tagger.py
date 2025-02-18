import boto3
import os

class BedrockTagger:
    def __init__(self):
        region_name = os.getenv('AWS_REGION', 'us-west-2')
        self.bedrock_client = boto3.client("bedrock", region_name=region_name)


    def create_inference_profile(self,profile_name, model_arn, tags):    
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

    def get_inference_profile_by_name(self,profile_name):
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

    def list_inference_profiles(self):

        return self.bedrock_client.list_inference_profiles()

    def delete_inference_profiles(self,profile_name):
        
        self.bedrock_client.delete_inference_profile(
            inferenceProfileIdentifier=profile_name)     




    

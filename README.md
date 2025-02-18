# AWS Bedrock Inference Profile Management Tool

## Introduction
This tool provides a comprehensive solution for managing AWS Bedrock Application Inference Profiles. It allows you to create, list, and manage application inference profiles for Foundation Models or Cross-region Inference Profiles. The tool includes tagging capabilities and provides an interactive command line interface for easy management.

## Features
- Create Application Inference Profiles with:
  - Foundation Model support
  - Cross-region Inference Profile support
- Created profiles are exported to CSV automatically
- Tag management for inference profiles
- List existing Application inference profiles
- Delete existing profiles
- Interactive command-line interface
- **Only support On-Demand Models, do not support Provisioned Models**.

## Prerequisites
- Python 3.9 and above
- AWS Account with appropriate permissions
- AWS credentials configured (either through AWS CLI profiles or access keys)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd tag_bedrock
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Required AWS Permissions
When using the Application Inference Profile, ensure you have the following IAM permissions (replace `<region>` and `<account_id>` with your values):

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["bedrock:InvokeModel*"],
      "Resource": [
        "arn:aws:bedrock:<region>:<account_id>:inference-profile/*",
        "arn:aws:bedrock:<region>::foundation-model/*"
      ]
    }
  ]
}
```

## Usage

### Creating a New Inference Profile
Run the tool in interactive creation mode:
```bash
python bedrock_inference_profile_management_tool.py
```

The tool will guide you through:
1. AWS credential configuration (profile selection or manual input)
2. Tag configuration
3. Profile creation with options for:
   - Foundation Models
   - Cross-region Inference Profiles

### Listing and Managing Existing Profiles
To list and manage existing profiles:
```bash
python bedrock_inference_profile_management_tool.py -l
```

This command will:
1. Display all existing Application inference profiles
2. Allow you to delete selected profiles
3. Show profile details including:
   - Profile Name
   - Region
   - Model ID
   - Status
   - ARN
   - Associated Tags

### CSV Export
The tool automatically exports profile information to CSV files with timestamps for record-keeping. The CSV includes:
- Profile Name
- Profile ARN
- Associated Tags

## Error Handling
- The tool includes comprehensive error handling for common scenarios
- Provides clear error messages and retry options
- Validates inputs to prevent invalid operations

## Contributing
Contributions are welcome! Please feel free to submit pull requests or create issues for bugs and feature requests.

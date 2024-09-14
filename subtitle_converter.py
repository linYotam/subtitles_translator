import os
import json
import boto3
import re
from botocore.exceptions import ClientError
import datetime
from botocore.config import Config

# Set up the AWS Bedrock client with increased timeout
config = Config(connect_timeout=300, read_timeout=300)  # Adjust as needed
client = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1',
    aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
    aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY'),
    config=config
)

# Set the model ID for Bedrock
model_id = "anthropic.claude-3-5-sonnet-20240620-v1:0"

# Directory paths
input_dir = os.path.join(os.getcwd(), "Input Subtitles")
output_dir = os.path.join(os.getcwd(), "Output Subtitles")

# Function to invoke AWS Bedrock for translation
def invoke_bedrock(model_id, prompt):
    try:
        translation_prompt = f"""Translate the following text to Hebrew but return the same structure. For example:
        12
        00:05:26,451 --> 00:05:30,247
        This place be four,
        five miles back,
        as you sit now.
        should be translated to:
        12
        00:05:26,451 --> 00:05:30,247
        המקום הזה יהיה ארבע,
        חמישה מייל אחורה,
        כפי שאתה יושב עכשיו.
        {prompt}"""

        request_body = json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2000,
            "temperature": 0.7,
            "messages": [
                {"role": "user", "content": translation_prompt}
            ]
        })

        response = client.invoke_model(
            modelId=model_id,
            body=request_body
        )

        # Read and print the response body for debugging
        response_body = json.loads(response['body'].read())
        print("Response Body:", response_body)

        # Extract the translated text from the response
        content = response_body.get('content', [])
        if content:
            # Assuming 'content' is a list of dictionaries and we need the 'text' field
            translated_text = ''.join(item.get('text', '') for item in content)
            return translated_text
        else:
            print("ERROR: 'content' key not found in response body.")
            return None

    except ClientError as e:
        print(f"ERROR: Can't invoke '{model_id}'. Reason: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Unexpected error. Reason: {e}")
        return None



# Function to read the .srt file
def read_srt(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Function to split the text into chunks based on token limits
def split_into_chunks(text, max_tokens=2000):
    # Splitting into chunks based on some sensible logic (such as paragraph or sentence boundaries)
    # For simplicity, this splits by line here. You can adjust to fit your requirements.
    lines = text.splitlines()
    chunks = []
    current_chunk = []
    current_length = 0

    for line in lines:
        current_length += len(line.split())  # Approximate token count by splitting on spaces
        if current_length > max_tokens:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            current_length = len(line.split())

        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

# Function to translate the SRT file in chunks
def translate_srt(input_path, output_path):
    original_text = read_srt(input_path)
    chunks = split_into_chunks(original_text)
    translated_chunks = []
    # Calculate the total number of chunks
    total_chunks = len(chunks)  
    print(f"Total Chunks: {total_chunks}")
    for index, chunk in enumerate(chunks):
        print(f"Translating chunk {index + 1} of {total_chunks} chunks, size: {len(chunk.split())} tokens")
                # Print the current time
        start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"Start time: {start_time}")
        translated_chunk = invoke_bedrock(model_id, chunk)
        if translated_chunk:
            translated_chunks.append(translated_chunk)
        else:
            translated_chunks.append("[Translation failed]")
        end_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"End time: {end_time}")    

    # Save the combined translated text to the output file
    with open(output_path, 'w', encoding='utf-8') as output_file:
        output_file.write("\n\n".join(translated_chunks))

# Main execution
if __name__ == "__main__":

    # Loop through all .srt files in the input directory
    for file_name in os.listdir(input_dir):
        if file_name.endswith('.srt'):
            input_srt_file = os.path.join(input_dir, file_name)
            
            # Generate output file name with _heb.srt
            output_file_name = os.path.splitext(file_name)[0] + "_heb.srt"
            output_srt_file = os.path.join(output_dir, output_file_name)
            
            print(f"Processing file: {file_name}")

            # Translate the SRT file
            translate_srt(input_srt_file, output_srt_file)

            print(f"Translation completed! Translated file saved at: {output_srt_file}")

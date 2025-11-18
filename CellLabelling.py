import SuperCellposeLoop
import SingleCell_Fromnpy

# # Run the Segmentation
# SuperCellposeLoop.run_segmentation()

# # Extract Single Cell Images
# SingleCell_Fromnpy.extract_single_cells()

# Define the URL where Label Studio is accessible
LABEL_STUDIO_URL = 'http://localhost:8080/'
# API key is available at the Account & Settings page in Label Studio UI
LABEL_STUDIO_API_KEY = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0b2tlbl90eXBlIjoicmVmcmVzaCIsImV4cCI6ODA3MDE4NzIzMywiaWF0IjoxNzYyOTg3MjMzLCJqdGkiOiJiZjNkZTRhNmE0OGM0NjZjODU2YjQ3MjRmZWQxNjhkNCIsInVzZXJfaWQiOiIxIn0.qIxLTlnSyG2WTKW4j7ouHXhkDZETpTBGaqZnBBw7ks8'

# Import the SDK and the client module
from label_studio_sdk import LabelStudio

# Connect to the Label Studio API 
client = LabelStudio(base_url=LABEL_STUDIO_URL, api_key=LABEL_STUDIO_API_KEY)

# A basic request to verify connection is working
me = client.users.whoami()

print("username:", me.username)
print("email:", me.email)

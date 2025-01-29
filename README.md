Step 1: Install Required Libraries

pip install qdrant-client sentence-transformers groq tiktoken python-dotenv PyPDF2 numpy

Step 2: Set Up Environment Variables

Create a .env file in your project directory and add the required API keys:

QDRANT_URL="your_qdrant_url"

QDRANT_API_KEY="your_qdrant_api_key"

GROQ_API_KEY="your_groq_api_key"

These keys are necessary for interacting with Qdrant (vector database) and Groq (LLM API).


Step 3: Create the Python Script

Save the provided script as app.py in your project directory.


Step 4: Run the Script

Execute the script with:

python app.py

If a PDF file named Data Science.pdf is in the same directory, it will be processed and indexed. You can then interactively ask questions.


Step 5: Testing and Validation

Make sure the .env file contains valid API keys.

Ensure Data Science.pdf exists for indexing.

Try different queries to verify the search and response generation.





1. Planned Technologies to Resolve
•	Lightweight NLP & Sentiment Analysis: Utilizing accessible, pre-trained AI models (e.g., Hugging Face's FinBERT or a basic LLM API like Gemini/OpenAI) to analyze financial text without requiring local GPU training.

•	Conceptual Text Extraction: Basic text parsing from digital PDFs or HTML announcements, skipping complex OCR for the sake of a workable demo.

2. Tools, Data Sources, and Libraries
•	Development Environment: Google Colab.

•	Data Sources: A curated, static sample dataset of 10-20 recent HKEX announcements (e.g., profit warnings, dividend declarations) downloaded manually or via a simple script. 

•	Libraries:
o	pdfplumber or BeautifulSoup for simple text extraction.
o	transformers (Hugging Face) for sentiment classification.
o	yfinance to fetch basic historical stock prices to show correlations.
o	Interactive UI directly within presentation demo.
o   Download  HKEX PDF announcements via https://www1.hkexnews.hk/search/titlesearch.xhtml?lang=en 


3. Process and Methodology
•	Phase 1: Data Setup: download real-world HKEX PDF announcements by entering stock number in interactive UI.

•	Phase 2: Extraction & Processing Script: Write Python code to extract text from a selected document and pass it to the NLP model to determine if the news is positive, negative, or neutral.

•	Phase 3: Interactive Demo Build: Wrap the Python script in a simple Gradio web interface hosted within Colab. This allows users (or the professor during the presentation) to select a document from a dropdown and instantly see the AI-generated summary.

•	Phase 4: Iterative Refinement: Following the Design Thinking methodology, test the Colab notebook with teammates to refine the code and ensure the demo runs smoothly without errors before the final presentation. 

4. Testing, Outcome Analysis, and KPIs
•	Testing Protocol: Execute the Colab notebook using a small set of historical announcements where the market outcome is already known, and analyze if the AI correctly flagged the event.
 
•	Evaluation Criteria (KPIs): 
o	Demo Stability: The Colab notebook runs end-to-end without library dependency errors.
o	Summary Accuracy: The AI successfully captures the core financial event in 3-5 bullet points.
o	Processing Speed: The script returns a result in under 15 seconds for a live demonstration.

•	Ethics & Limitations Analysis: The final report will discuss the limitations of a simplified demo model (e.g., hallucination risks, parsing errors) and the ethical implications of relying on unverified AI outputs for actual retail investing. 

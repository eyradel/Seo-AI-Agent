# AI SEO Agent

A powerful SEO analysis tool that leverages AI to provide comprehensive website audits and actionable recommendations.

## Description

The AI SEO Agent automatically crawls websites to analyze key SEO factors including:

- Technical SEO elements (meta tags, headings, schema markup)
- Content quality and keyword optimization
- Page performance and mobile-friendliness
- Potential issues and improvement opportunities

The tool generates a prioritized action plan with specific recommendations to improve your SEO performance. The interactive dashboard provides visual reporting and detailed breakdowns of all analyzed metrics.

## Installation

1. Clone the repository or download the files
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Launch the app using Streamlit:

```bash
streamlit run app.py
```

The application will open in your default web browser (typically at http://localhost:8501).

## Usage

1. Enter your website URL
2. Add target keywords (separated by commas)
3. Adjust the analysis depth and page limit if needed
4. Click "Start Analysis"
5. Review your SEO audit results across the different tabs

## API Keys (Optional)

For enhanced functionality, you can add API keys in your environment variables:

- OpenAI API key for content analysis
- Google API key for SERP competition analysis
- Gemini API key as an alternative for content analysis

The app will work with basic SEO analysis even without these API keys.
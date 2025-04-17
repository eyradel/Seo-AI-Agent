import streamlit as st
import requests
from urllib.parse import urlparse
import re
import time
import pandas as pd
from datetime import datetime
import json
import openai
from googleapiclient.discovery import build
import os
import concurrent.futures
from bs4 import BeautifulSoup
from google import genai


# Set API keys (in a production environment, use environment variables or secure storage)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"] if "OPENAI_API_KEY" in st.secrets else os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] if "GOOGLE_API_KEY" in st.secrets else os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"] if "GEMINI_API_KEY" in st.secrets else os.environ.get("GEMINI_API_KEY", "")



# UI Configuration
def load_css():
    st.markdown(
        """
        <meta charset="UTF-8">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/mdbootstrap/4.19.1/css/mdb.min.css" rel="stylesheet">
        <link href="https://cdnjs.cloudflare.com/ajax/libs/bootstrap/5.3.2/css/bootstrap.min.css" rel="stylesheet">
        <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
        
        <title>AI SEO Agent</title>
        <meta name="title" content="AI SEO Agent" />
        <meta name="description" content="Autonomous SEO analysis and recommendations" />
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
            header {visibility: hidden;}
            .main {
                margin-top: -20px;
                padding-top: 10px;
            }
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            .navbar {
                padding: 1rem;
                margin-bottom: 2rem;
                background-color: #4267B2;
                color: white;
            }
            .card {
                padding: 1rem;
                margin-bottom: 1rem;
                transition: transform 0.2s;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
            .card:hover {
                transform: scale(1.02);
            }
            .metric-card {
                border-radius: 8px;
                padding: 1rem;
                margin: 0.5rem;
            }
            .search-box {
                margin-bottom: 1rem;
                padding: 0.5rem;
                border-radius: 4px;
            }
            .agent-message {
                background-color: #f0f2f5;
                padding: 12px;
                border-radius: 10px;
                margin-bottom: 10px;
            }
            .chat-container {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 10px;
                height: 300px;
                overflow-y: auto;
                margin-bottom: 10px;
                background-color: white;
            }
            .analysis-container {
                border: 1px solid #e0e0e0;
                border-radius: 10px;
                padding: 20px;
                margin-bottom: 20px;
            }
            .progress-label {
                font-size: 14px;
                color: #555;
                margin-bottom: 5px;
            }
            .priority-high {
                border-left: 4px solid #dc3545;
                padding-left: 10px;
            }
            .priority-medium {
                border-left: 4px solid #ffc107;
                padding-left: 10px;
            }
            .priority-low {
                border-left: 4px solid #28a745;
                padding-left: 10px;
            }
            .step-card {
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 8px;
                margin-bottom: 15px;
                border-top: 3px solid #4267B2;
            }
            .tag {
                display: inline-block;
                padding: 3px 8px;
                background-color: #e0e0e0;
                border-radius: 12px;
                font-size: 12px;
                margin-right: 5px;
                margin-bottom: 5px;
            }
            .tag-success {
                background-color: #d4edda;
                color: #155724;
            }
            .tag-warning {
                background-color: #fff3cd;
                color: #856404;
            }
            .tag-danger {
                background-color: #f8d7da;
                color: #721c24;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def create_navbar():
    st.markdown(
        """
       <nav class="navbar fixed-top navbar-expand-lg navbar-dark text-bold shadow-sm" >
            <a class="navbar-brand" href="#" target="_blank">
                <img src=""  style='width:50px'>
               AI SEO Agent 🤖
            </a>
        </nav>
        """,
        unsafe_allow_html=True
    )

# SEO Agent Class
class SEOAgent:
    def __init__(self, openai_key=None, google_key=None, gemini_key=None):
        self.openai_api_key = openai_key
        self.google_api_key = google_key
        self.gemini_api_key = gemini_key
        self.chat_history = []
        self.analysis_results = {}
        self.recommendations = {}
        self.insights = {}
        self.competitor_data = {}
        self.status = "idle"
        
        # Configure OpenAI if key is provided
        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Set up agent capabilities
        self.capabilities = {
            "technical_seo": True,
            "content_analysis": bool(self.openai_api_key or self.gemini_api_key),
            "competitor_analysis": bool(self.google_api_key),
            "keyword_research": bool(self.google_api_key or self.openai_api_key or self.gemini_api_key)
        }
    
    def add_message(self, role, content):
        """Add a message to the chat history"""
        self.chat_history.append({"role": role, "content": content, "timestamp": datetime.now()})
        return len(self.chat_history) - 1
    
    def analyze_website(self, url, keywords=None, max_pages=5, analysis_depth="Standard"):
        """Main function to perform website analysis"""
        self.status = "analyzing"
        self.add_message("agent", f"Starting analysis of {url}. This may take a few minutes...")
        
        # Basic URL validation
        if not url.startswith(('http://', 'https://')):
            url = 'https://' + url
        
        # Setup for crawling
        base_domain = urlparse(url).netloc
        results = self._crawl_site(url, max_pages, base_domain)
        
        # Process keywords
        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(",")]
            self.add_message("agent", f"Analyzing keyword presence for: {', '.join(keyword_list)}")
        
        # Enhanced analysis with AI if available
        if self.capabilities["content_analysis"] and analysis_depth != "Basic":
            self._enhance_with_ai(results, keyword_list, analysis_depth)
        
        # Competitor analysis if available and requested
        if self.capabilities["competitor_analysis"] and keyword_list and analysis_depth == "Deep":
            self._analyze_competitors(base_domain, keyword_list)
        
        # Generate insights and recommendations
        self._generate_insights(results, keyword_list, base_domain)
        
        # Store results
        self.analysis_results = {
            "url": url,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pages": results,
            "keywords": keyword_list,
            "domain": base_domain
        }
        
        self.status = "complete"
        self.add_message("agent", "Analysis complete! I've generated insights and recommendations for your website.")
        
        return results
    
    def _crawl_site(self, start_url, max_pages, base_domain):
        """Crawl the website and analyze pages"""
        results = []
        to_crawl = {start_url}
        crawled = set()
        
        self.add_message("agent", f"Crawling website starting from {start_url}...")
        
        # Crawl pages
        while to_crawl and len(crawled) < max_pages:
            current_url = to_crawl.pop()
            if current_url in crawled:
                continue
            
            self.add_message("agent", f"Analyzing page: {current_url}")
            
            try:
                # Get page content
                start_time = time.time()
                response = requests.get(current_url, timeout=15)
                load_time = time.time() - start_time
                
                if response.status_code == 200:
                    html = response.text
                    
                    # Extract page data
                    page_data = self._extract_page_data(html, current_url, load_time)
                    
                    # Extract links for further crawling
                    internal_links = self._extract_links(html, current_url, base_domain)
                    for link in internal_links:
                        if link not in crawled and len(to_crawl) + len(crawled) < max_pages:
                            to_crawl.add(link)
                    
                    results.append(page_data)
                else:
                    self.add_message("agent", f"⚠️ Warning: Couldn't access {current_url} (Status code: {response.status_code})")
                
                crawled.add(current_url)
                
            except Exception as e:
                self.add_message("agent", f"⚠️ Error analyzing {current_url}: {str(e)}")
                crawled.add(current_url)
        
        self.add_message("agent", f"Finished crawling. Analyzed {len(results)} pages.")
        return results
    
    def _extract_page_data(self, html, url, load_time):
        """Extract relevant SEO data from a page"""
        soup = BeautifulSoup(html, 'html.parser')
        
        # Title
        title = soup.title.text if soup.title else "No title found"
        
        # Meta description
        meta_desc_tag = soup.find('meta', attrs={'name': 'description'})
        description = meta_desc_tag.get('content') if meta_desc_tag else "No description found"
        
        # Clean text content
        for script in soup(["script", "style"]):
            script.extract()
        text_content = soup.get_text(separator=' ')
        text_content = re.sub(r'\s+', ' ', text_content).strip()
        
        # Get a sample of the text for AI analysis
        text_sample = text_content[:5000] if len(text_content) > 5000 else text_content
        
        # Word count
        word_count = len(re.findall(r'\b\w+\b', text_content))
        
        # Headings
        h1_tags = [h1.text.strip() for h1 in soup.find_all('h1')]
        h2_tags = [h2.text.strip() for h2 in soup.find_all('h2')]
        h3_tags = [h3.text.strip() for h3 in soup.find_all('h3')]
        
        # Images
        img_tags = soup.find_all('img')
        img_count = len(img_tags)
        img_missing_alt = sum(1 for img in img_tags if not img.get('alt'))
        
        # Schema markup
        schema_tags = soup.find_all('script', type='application/ld+json')
        has_schema = len(schema_tags) > 0
        
        # Mobile friendliness signals
        viewport_tag = soup.find('meta', attrs={'name': 'viewport'})
        has_viewport = bool(viewport_tag)
        
        # Calculate page size (approximation)
        page_size = len(html) / 1024  # in KB
        
        # Calculate SEO score
        score, issues = self._calculate_seo_score({
            'title': title,
            'title_length': len(title),
            'description': description,
            'description_length': len(description) if description != "No description found" else 0,
            'word_count': word_count,
            'h1_count': len(h1_tags),
            'img_count': img_count,
            'img_missing_alt': img_missing_alt,
            'load_time': load_time,
            'has_schema': has_schema,
            'has_viewport': has_viewport,
            'page_size': page_size
        })
        
        return {
            'url': url,
            'title': title,
            'title_length': len(title),
            'description': description,
            'description_length': len(description) if description != "No description found" else 0,
            'text_sample': text_sample,
            'word_count': word_count,
            'h1_tags': h1_tags,
            'h2_tags': h2_tags,
            'h3_tags': h3_tags,
            'h1_count': len(h1_tags),
            'h2_count': len(h2_tags),
            'h3_count': len(h3_tags),
            'img_count': img_count,
            'img_missing_alt': img_missing_alt,
            'has_schema': has_schema,
            'has_viewport': has_viewport,
            'load_time': load_time,
            'page_size': page_size,
            'score': score,
            'issues': issues
        }
    
    def _extract_links(self, html, current_url, base_domain):
        """Extract internal links from HTML"""
        soup = BeautifulSoup(html, 'html.parser')
        internal_links = []
        
        for a_tag in soup.find_all('a', href=True):
            link = a_tag['href']
            
            # Skip anchors and javascript
            if link.startswith('#') or link.startswith('javascript:'):
                continue
                
            # Convert relative URLs to absolute
            if link.startswith('/'):
                full_url = f"{urlparse(current_url).scheme}://{base_domain}{link}"
                internal_links.append(full_url)
            elif base_domain in link:
                internal_links.append(link)
        
        return internal_links
    
    def _calculate_seo_score(self, page_data):
        """Calculate SEO score and identify issues"""
        score = 100
        issues = []
        
        # Title checks
        if page_data['title'] == "No title found":
            score -= 20
            issues.append({"type": "title", "severity": "high", "message": "Missing title tag"})
        elif page_data['title_length'] < 30:
            score -= 10
            issues.append({"type": "title", "severity": "medium", "message": "Title too short (under 30 characters)"})
        elif page_data['title_length'] > 60:
            score -= 5
            issues.append({"type": "title", "severity": "low", "message": "Title too long (over 60 characters)"})
        
        # Meta description checks
        if page_data['description'] == "No description found":
            score -= 15
            issues.append({"type": "meta", "severity": "high", "message": "Missing meta description"})
        elif page_data['description_length'] < 80:
            score -= 10
            issues.append({"type": "meta", "severity": "medium", "message": "Meta description too short (under 80 characters)"})
        elif page_data['description_length'] > 160:
            score -= 5
            issues.append({"type": "meta", "severity": "low", "message": "Meta description too long (over 160 characters)"})
        
        # Content checks
        if page_data['word_count'] < 300:
            score -= 15
            issues.append({"type": "content", "severity": "high", "message": "Low word count (under 300 words)"})
        elif page_data['word_count'] < 600:
            score -= 5
            issues.append({"type": "content", "severity": "low", "message": "Content could be more comprehensive (under 600 words)"})
        
        # H1 checks
        if page_data['h1_count'] == 0:
            score -= 10
            issues.append({"type": "headings", "severity": "high", "message": "Missing H1 heading"})
        elif page_data['h1_count'] > 1:
            score -= 5
            issues.append({"type": "headings", "severity": "medium", "message": f"Multiple H1 headings ({page_data['h1_count']})"})
        
        # Image checks
        if page_data['img_count'] > 0 and page_data['img_missing_alt'] > 0:
            score -= min(10, page_data['img_missing_alt'] * 2)
            percent_missing = (page_data['img_missing_alt'] / page_data['img_count']) * 100
            severity = "high" if percent_missing > 50 else "medium"
            issues.append({"type": "images", "severity": severity, "message": f"{page_data['img_missing_alt']} images missing alt text"})
        
        # Load time checks
        if page_data['load_time'] > 5:
            score -= 15
            issues.append({"type": "performance", "severity": "high", "message": f"Very slow load time ({page_data['load_time']:.2f}s)"})
        elif page_data['load_time'] > 3:
            score -= 10
            issues.append({"type": "performance", "severity": "medium", "message": f"Slow load time ({page_data['load_time']:.2f}s)"})
        
        # Schema checks
        if not page_data['has_schema']:
            score -= 5
            issues.append({"type": "schema", "severity": "medium", "message": "No structured data (Schema.org) found"})
        
        # Mobile friendliness
        if not page_data['has_viewport']:
            score -= 10
            issues.append({"type": "mobile", "severity": "high", "message": "No viewport meta tag (not mobile-friendly)"})
        
        # Page size
        if page_data['page_size'] > 5000:  # 5MB
            score -= 10
            issues.append({"type": "performance", "severity": "high", "message": f"Page size too large ({page_data['page_size']/1024:.1f} MB)"})
        elif page_data['page_size'] > 2000:  # 2MB
            score -= 5
            issues.append({"type": "performance", "severity": "medium", "message": f"Page size large ({page_data['page_size']/1024:.1f} MB)"})
        
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        return score, issues
    
    def _enhance_with_ai(self, results, keyword_list, depth):
        """Enhance analysis with AI content scoring"""
        if not self.openai_api_key and not self.gemini_api_key:
            self.add_message("agent", "⚠️ AI content analysis unavailable. No API keys provided.")
            return
        
        self.add_message("agent", "Performing AI content analysis...")
        
        for i, result in enumerate(results):
            self.add_message("agent", f"Analyzing content quality for: {result['url']}")
            
            if self.openai_api_key:
                result['ai_analysis'] = self._analyze_with_openai(result['title'], result['text_sample'], keyword_list)
            elif self.gemini_api_key:
                result['ai_analysis'] = self._analyze_with_gemini(result['title'], result['text_sample'], keyword_list)
            
            # Add keyword analysis if keywords provided
            if keyword_list:
                result['keyword_presence'] = self._analyze_keyword_presence(
                    result['title'], 
                    result['description'], 
                    result['h1_tags'] + result['h2_tags'], 
                    result['text_sample'], 
                    keyword_list
                )
    
    def _analyze_with_openai(self, title, content, keywords=None):
        """Analyze content quality using OpenAI API"""
        try:
            # Prepare prompt
            prompt = f"""
            Analyze this webpage content from an SEO perspective:
            
            TITLE: {title}
            
            CONTENT SAMPLE: {content[:1500]}...
            """
            
            if keywords:
                prompt += f"\nTARGET KEYWORDS: {', '.join(keywords)}\n\n"
                
            prompt += """
            Please provide the following analysis in JSON format:
            1. content_quality_score: A score from 0-100 evaluating the content quality
            2. readability_score: A score from 0-100 evaluating how readable the content is
            3. keyword_usage: How well keywords are used in the content (excellent/good/fair/poor)
            4. strengths: An array of content strengths (max 3 points)
            5. weaknesses: An array of content weaknesses (max 3 points)
            6. recommendations: An array of specific content improvement recommendations (max 3 points)
            
            Return only valid JSON with these exact keys.
            """
            
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={ "type": "json_object" },
                messages=[
                    {"role": "system", "content": "You are an expert SEO consultant analyzing web content."},
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse response
            ai_analysis = json.loads(response.choices[0].message.content)
            return ai_analysis
            
        except Exception as e:
            self.add_message("agent", f"⚠️ Error in OpenAI analysis: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_with_gemini(self, title, content, keywords=None):
        """Analyze content quality using Google's Gemini API"""
        try:
           
            client = genai.Client(api_key=GEMINI_API_KEY)
            # Prepare prompt
            prompt = f"""
            Analyze this webpage content from an SEO perspective:
            
            TITLE: {title}
            
            CONTENT SAMPLE: {content[:1500]}...
            """
            
            if keywords:
                prompt += f"\nTARGET KEYWORDS: {', '.join(keywords)}\n\n"
                
            prompt += """
            Please provide the following analysis in JSON format:
            1. content_quality_score: A score from 0-100 evaluating the content quality
            2. readability_score: A score from 0-100 evaluating how readable the content is
            3. keyword_usage: How well keywords are used in the content (excellent/good/fair/poor)
            4. strengths: An array of content strengths (max 3 points)
            5. weaknesses: An array of content weaknesses (max 3 points)
            6. recommendations: An array of specific content improvement recommendations (max 3 points)
            
            Return only valid JSON with these exact keys.
            """
            
            response = client.models.generate_content(
                    model="gemini-2.0-flash",
                    contents=prompt
                )
            
            # Parse response
            response_text = response.text
            # Extract JSON from response (in case there's additional text)
            json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
            if json_str:
                ai_analysis = json.loads(json_str.group(0))
            else:
                ai_analysis = {"error": "Could not parse JSON from Gemini response"}
            
            return ai_analysis
            
        except Exception as e:
            self.add_message("agent", f"⚠️ Error in Gemini analysis: {str(e)}")
            return {"error": str(e)}
    
    def _analyze_keyword_presence(self, title, description, headings, content, keywords):
        """Analyze keyword presence in page elements"""
        keyword_presence = {}
        
        for keyword in keywords:
            keyword_lower = keyword.lower()
            in_title = keyword_lower in title.lower()
            in_description = description != "No description found" and keyword_lower in description.lower()
            in_headings = any(keyword_lower in h.lower() for h in headings)
            
            # Count occurrences in content
            occurrences = content.lower().count(keyword_lower)
            
            # Calculate keyword density
            word_count = len(re.findall(r'\b\w+\b', content))
            density = (occurrences / word_count * 100) if word_count > 0 else 0
            
            keyword_presence[keyword] = {
                'in_title': in_title,
                'in_description': in_description,
                'in_headings': in_headings,
                'occurrences': occurrences,
                'density': density
            }
        
        return keyword_presence
    
    def _analyze_competitors(self, domain, keywords):
        """Analyze search competition for keywords"""
        if not self.google_api_key:
            self.add_message("agent", "⚠️ Competitor analysis unavailable. No Google API key provided.")
            return
        
        self.add_message("agent", "Analyzing search competition...")
        
        serp_data = {}
        for keyword in keywords:
            self.add_message("agent", f"Checking SERP for: {keyword}")
            serp_data[keyword] = self._get_serp_data(keyword, domain)
        
        self.competitor_data = serp_data
    
    def _get_serp_data(self, keyword, domain):
        """Get SERP data for a keyword"""
        try:
            service = build(
                "customsearch", "v1",
                developerKey=self.google_api_key
            )
            
            result = service.cse().list(
                q=keyword,
                cx='YOUR_CUSTOM_SEARCH_ENGINE_ID',  # Custom Search Engine ID
                num=10
            ).execute()
            
            # Parse results
            serp_data = {
                "total_results": result.get("searchInformation", {}).get("totalResults", 0),
                "domain_position": None,
                "top_results": []
            }
            
            # Process results
            if "items" in result:
                for i, item in enumerate(result["items"]):
                    result_domain = urlparse(item["link"]).netloc
                    
                    # Check if this is our domain
                    if domain in result_domain:
                        serp_data["domain_position"] = i + 1
                    
                    # Add to top results
                    serp_data["top_results"].append({
                        "position": i + 1,
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "domain": result_domain,
                        "snippet": item.get("snippet", ""),
                        "is_our_domain": domain in result_domain
                    })
            
            return serp_data
            
        except Exception as e:
            self.add_message("agent", f"⚠️ Error retrieving SERP data: {str(e)}")
            return {"error": str(e)}
    
    def _generate_insights(self, results, keywords, domain):
        """Generate insights and recommendations from analysis results"""
        self.add_message("agent", "Generating insights and recommendations...")
        
        # Calculate aggregate metrics
        avg_score = sum(page['score'] for page in results) / len(results)
        avg_word_count = sum(page['word_count'] for page in results) / len(results)
        avg_load_time = sum(page['load_time'] for page in results) / len(results)
        
        # Collect all issues
        all_issues = []
        for page in results:
            for issue in page['issues']:
                all_issues.append({
                    'url': page['url'],
                    'issue': issue
                })
        
        # Count issue types
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue['issue']['type']
            if issue_type in issue_counts:
                issue_counts[issue_type] += 1
            else:
                issue_counts[issue_type] = 1
        
        # Generate site-wide insights
        self.insights = {
            "overall_score": avg_score,
            "grade": self._get_grade(avg_score),
            "avg_word_count": avg_word_count,
            "avg_load_time": avg_load_time,
            "top_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True),
            "keyword_presence": self._analyze_site_keyword_presence(results, keywords) if keywords else None,
        }
        
        # Generate recommendations
        self._generate_recommendations(results, keywords, domain)
    
    def _analyze_site_keyword_presence(self, results, keywords):
        """Analyze keyword presence across the site"""
        site_keyword_data = {}
        
        for keyword in keywords:
            present_in_title = sum(1 for r in results if 'keyword_presence' in r and keyword in r['keyword_presence'] and r['keyword_presence'][keyword]['in_title'])
            present_in_desc = sum(1 for r in results if 'keyword_presence' in r and keyword in r['keyword_presence'] and r['keyword_presence'][keyword]['in_description'])
            present_in_h = sum(1 for r in results if 'keyword_presence' in r and keyword in r['keyword_presence'] and r['keyword_presence'][keyword]['in_headings'])
            
            total_occurrences = sum(r['keyword_presence'][keyword]['occurrences'] for r in results if 'keyword_presence' in r and keyword in r['keyword_presence'])
            
            avg_density = sum(r['keyword_presence'][keyword]['density'] for r in results if 'keyword_presence' in r and keyword in r['keyword_presence']) / len(results) if len(results) > 0 else 0
            
            site_keyword_data[keyword] = {
                'in_title_count': present_in_title,
                'in_title_percentage': (present_in_title / len(results) * 100) if len(results) > 0 else 0,
                'in_description_count': present_in_desc,
                'in_description_percentage': (present_in_desc / len(results) * 100) if len(results) > 0 else 0,
                'in_headings_count': present_in_h,
                'in_headings_percentage': (present_in_h / len(results) * 100) if len(results) > 0 else 0,
                'total_occurrences': total_occurrences,
                'avg_density': avg_density
            }
        
        return site_keyword_data
    
    def _generate_recommendations(self, results, keywords, domain):
        """Generate actionable recommendations based on analysis"""
        recommendations = {
            "critical": [],
            "important": [],
            "opportunity": []
        }
        
        # Technical SEO recommendations
        self._add_technical_recommendations(results, recommendations)
        
        # Content recommendations
        self._add_content_recommendations(results, keywords, recommendations)
        
        # Performance recommendations
        self._add_performance_recommendations(results, recommendations)
        
        # Mobile-friendliness recommendations
        self._add_mobile_recommendations(results, recommendations)
        
        # Add AI recommendations if available
        self._add_ai_recommendations(results, recommendations)
        
        # Competitor-based recommendations if available
        if self.competitor_data and keywords:
            self._add_competitor_recommendations(recommendations, keywords, domain)
        
        self.recommendations = recommendations
    
    def _add_technical_recommendations(self, results, recommendations):
        """Add technical SEO recommendations"""
        # Title tag issues
        missing_titles = [p['url'] for p in results if p.get('title') == "No title found"]
        if missing_titles:
            recommendations["critical"].append({
                "type": "title",
                "message": f"Add title tags to {len(missing_titles)} pages",
                "details": "Title tags are critical for SEO and user experience",
                "affected_pages": missing_titles
            })
        
        # Meta description issues
        missing_desc = [p['url'] for p in results if p.get('description') == "No description found"]
        if missing_desc:
            recommendations["important"].append({
                "type": "meta",
                "message": f"Add meta descriptions to {len(missing_desc)} pages",
                "details": "Meta descriptions improve click-through rates from search results",
                "affected_pages": missing_desc
            })
        
        # Heading structure issues
        missing_h1 = [p['url'] for p in results if p.get('h1_count', 0) == 0]
        if missing_h1:
            recommendations["important"].append({
                "type": "headings",
                "message": f"Add H1 headings to {len(missing_h1)} pages",
                "details": "Every page should have a single H1 heading that clearly describes the content",
                "affected_pages": missing_h1
            })
        
        multiple_h1 = [p['url'] for p in results if p.get('h1_count', 0) > 1]
        if multiple_h1:
            recommendations["important"].append({
                "type": "headings",
                "message": f"Fix multiple H1 headings on {len(multiple_h1)} pages",
                "details": "Each page should have exactly one H1 heading for proper hierarchy",
                "affected_pages": multiple_h1
            })
        
        # Schema markup
        missing_schema = [p['url'] for p in results if not p.get('has_schema', False)]
        if missing_schema and len(missing_schema) > len(results) / 2:  # If more than half are missing schema
            recommendations["opportunity"].append({
                "type": "schema",
                "message": "Implement structured data markup (Schema.org)",
                "details": "Structured data helps search engines understand your content and can enable rich results",
                "affected_pages": missing_schema
            })
        
        # Image optimization
        pages_with_missing_alt = [p for p in results if p.get('img_count', 0) > 0 and p.get('img_missing_alt', 0) > 0]
        if pages_with_missing_alt:
            total_missing = sum(p.get('img_missing_alt', 0) for p in pages_with_missing_alt)
            recommendations["important"].append({
                "type": "images",
                "message": f"Add alt text to {total_missing} images across {len(pages_with_missing_alt)} pages",
                "details": "Alt text improves accessibility and helps search engines understand images",
                "affected_pages": [p['url'] for p in pages_with_missing_alt]
            })
    
    def _add_content_recommendations(self, results, keywords, recommendations):
        """Add content-related recommendations"""
        # Thin content pages
        thin_content = [p for p in results if p.get('word_count', 0) < 300]
        if thin_content:
            recommendations["important"].append({
                "type": "content",
                "message": f"Expand content on {len(thin_content)} pages with thin content",
                "details": "Pages with less than 300 words may be considered thin content by search engines",
                "affected_pages": [p['url'] for p in thin_content]
            })
        
        # Keyword usage recommendations
        if keywords and any('keyword_presence' in p for p in results):
            for keyword in keywords:
                # Check if keyword is rarely used in titles
                title_usage = sum(1 for p in results if 'keyword_presence' in p and keyword in p['keyword_presence'] and p['keyword_presence'][keyword]['in_title'])
                if title_usage < len(results) * 0.3:  # Less than 30% of pages
                    recommendations["important"].append({
                        "type": "keywords",
                        "message": f"Include keyword '{keyword}' in more page titles",
                        "details": "Keywords in titles have high SEO value and impact click-through rates",
                        "keyword": keyword
                    })
                
                # Check for keyword density issues
                low_density_pages = [p for p in results if 'keyword_presence' in p and keyword in p['keyword_presence'] and p['keyword_presence'][keyword]['density'] < 0.5]
                if low_density_pages and len(low_density_pages) > len(results) * 0.5:  # More than 50% of pages
                    recommendations["opportunity"].append({
                        "type": "keywords",
                        "message": f"Increase keyword '{keyword}' usage across the site",
                        "details": "Aim for a keyword density of 0.5-2% for better topical relevance",
                        "keyword": keyword
                    })
                
                high_density_pages = [p for p in results if 'keyword_presence' in p and keyword in p['keyword_presence'] and p['keyword_presence'][keyword]['density'] > 3]
                if high_density_pages:
                    recommendations["important"].append({
                        "type": "keywords",
                        "message": f"Reduce overuse of keyword '{keyword}' on {len(high_density_pages)} pages",
                        "details": "Keyword stuffing (density over 3%) can lead to penalties",
                        "affected_pages": [p['url'] for p in high_density_pages]
                    })
        
        # Content quality recommendations based on word count distribution
        total_words = sum(p.get('word_count', 0) for p in results)
        avg_words = total_words / len(results) if results else 0
        
        if avg_words < 500:
            recommendations["important"].append({
                "type": "content",
                "message": "Increase overall content depth across the site",
                "details": f"Current average word count is {int(avg_words)}. Aim for at least 700-1000 words per page for better topical coverage."
            })
    
    def _add_performance_recommendations(self, results, recommendations):
        """Add performance-related recommendations"""
        # Slow loading pages
        slow_pages = [p for p in results if p.get('load_time', 0) > 3]
        if slow_pages:
            recommendations["important"].append({
                "type": "performance",
                "message": f"Improve load time on {len(slow_pages)} slow pages",
                "details": "Pages should load in under 3 seconds for good user experience and SEO",
                "affected_pages": [p['url'] for p in slow_pages]
            })
        
        # Large page size
        large_pages = [p for p in results if p.get('page_size', 0) > 2000]  # Over 2MB
        if large_pages:
            recommendations["important"].append({
                "type": "performance",
                "message": f"Reduce page size on {len(large_pages)} large pages",
                "details": "Large pages slow down loading and hurt mobile user experience",
                "affected_pages": [p['url'] for p in large_pages]
            })
    
    def _add_mobile_recommendations(self, results, recommendations):
        """Add mobile-friendliness recommendations"""
        # Missing viewport
        no_viewport = [p['url'] for p in results if not p.get('has_viewport', False)]
        if no_viewport:
            recommendations["critical"].append({
                "type": "mobile",
                "message": f"Add viewport meta tags to {len(no_viewport)} pages",
                "details": "Mobile-friendliness is a ranking factor. Viewport meta tag is required for responsive design.",
                "affected_pages": no_viewport
            })
    
    def _add_ai_recommendations(self, results, recommendations):
        """Add AI-generated recommendations"""
        # Collect all AI recommendations
        ai_recommendations = []
        for page in results:
            if 'ai_analysis' in page and 'recommendations' in page['ai_analysis']:
                for rec in page['ai_analysis']['recommendations']:
                    ai_recommendations.append({
                        "url": page['url'],
                        "recommendation": rec
                    })
        
        # Find common patterns in AI recommendations
        if ai_recommendations:
            # Group similar recommendations
            content_quality_recs = [r for r in ai_recommendations if any(kw in r['recommendation'].lower() for kw in ['quality', 'depth', 'value'])]
            readability_recs = [r for r in ai_recommendations if any(kw in r['recommendation'].lower() for kw in ['readability', 'simplify', 'clarity'])]
            structure_recs = [r for r in ai_recommendations if any(kw in r['recommendation'].lower() for kw in ['structure', 'format', 'organize'])]
            
            if content_quality_recs and len(content_quality_recs) > len(results) * 0.3:  # More than 30% of pages
                recommendations["important"].append({
                    "type": "ai_content",
                    "message": "Improve overall content quality and depth",
                    "details": "AI analysis suggests that content quality could be improved across multiple pages",
                    "affected_pages": [r['url'] for r in content_quality_recs]
                })
            
            if readability_recs and len(readability_recs) > len(results) * 0.3:
                recommendations["important"].append({
                    "type": "ai_readability",
                    "message": "Improve content readability site-wide",
                    "details": "AI analysis suggests that content could be more readable and accessible",
                    "affected_pages": [r['url'] for r in readability_recs]
                })
            
            if structure_recs and len(structure_recs) > len(results) * 0.3:
                recommendations["opportunity"].append({
                    "type": "ai_structure",
                    "message": "Improve content structure and organization",
                    "details": "AI analysis suggests better formatting and structure would improve user experience",
                    "affected_pages": [r['url'] for r in structure_recs]
                })
    
    def _add_competitor_recommendations(self, recommendations, keywords, domain):
        """Add competitor-based recommendations"""
        for keyword in keywords:
            if keyword in self.competitor_data:
                serp_data = self.competitor_data[keyword]
                
                if "error" not in serp_data:
                    # Check our domain position
                    if serp_data.get('domain_position') is None:
                        recommendations["opportunity"].append({
                            "type": "competition",
                            "message": f"Create targeted content for keyword '{keyword}'",
                            "details": "Your domain is not ranking in the top 10 for this keyword",
                            "keyword": keyword
                        })
                    elif serp_data.get('domain_position', 0) > 3:
                        recommendations["opportunity"].append({
                            "type": "competition",
                            "message": f"Optimize existing content for keyword '{keyword}'",
                            "details": f"Your domain is ranked #{serp_data['domain_position']} for this keyword",
                            "keyword": keyword
                        })
                    
                    # Analyze top competitors for this keyword
                    if 'top_results' in serp_data and len(serp_data['top_results']) > 0:
                        # Extract competitor domains
                        competitor_domains = [result['domain'] for result in serp_data['top_results'] 
                                             if not result.get('is_our_domain', False)][:3]
                        
                        if competitor_domains:
                            recommendations["opportunity"].append({
                                "type": "competition",
                                "message": f"Analyze top competitors for keyword '{keyword}'",
                                "details": f"Study content from {', '.join(competitor_domains)} to improve your rankings",
                                "keyword": keyword,
                                "competitors": competitor_domains
                            })
    
    def _get_grade(self, score):
        """Convert score to letter grade"""
        if score >= 90:
            return "A"
        elif score >= 80:
            return "B"
        elif score >= 70:
            return "C"
        elif score >= 60:
            return "D"
        else:
            return "F"
    
    def get_action_plan(self):
        """Generate a prioritized action plan from recommendations"""
        if not self.recommendations:
            return []
        
        action_plan = []
        
        # First add all critical recommendations
        for rec in self.recommendations.get("critical", []):
            action_plan.append({
                "priority": "high",
                "task": rec["message"],
                "details": rec["details"],
                "type": rec["type"]
            })
        
        # Then add important recommendations
        for rec in self.recommendations.get("important", []):
            action_plan.append({
                "priority": "medium",
                "task": rec["message"],
                "details": rec["details"],
                "type": rec["type"]
            })
        
        # Finally add opportunity recommendations (limited to top 3)
        opportunity_recs = self.recommendations.get("opportunity", [])
        for rec in opportunity_recs[:3]:
            action_plan.append({
                "priority": "low",
                "task": rec["message"],
                "details": rec["details"],
                "type": rec["type"]
            })
        
        return action_plan
    
    def generate_seo_summary(self):
        """Generate a concise SEO summary for the website"""
        if not self.analysis_results or not self.insights:
            return "No analysis data available. Run analysis first."
        
        try:
            # Try to generate summary with OpenAI if available
            if self.openai_api_key:
                return self._generate_summary_with_openai()
            elif self.gemini_api_key:
                return self._generate_summary_with_gemini()
            else:
                # Generate a generic summary
                return self._generate_generic_summary()
        except Exception as e:
            self.add_message("agent", f"⚠️ Error generating summary: {str(e)}")
            return self._generate_generic_summary()
    
    def _generate_summary_with_openai(self):
        """Generate summary using OpenAI"""
        client = openai.OpenAI(api_key=self.openai_api_key)
        
        # Prepare summary data
        summary_data = {
            "website": self.analysis_results["url"],
            "overall_score": int(self.insights["overall_score"]),
            "grade": self.insights["grade"],
            "analyzed_pages": len(self.analysis_results["pages"]),
            "avg_word_count": int(self.insights["avg_word_count"]),
            "avg_load_time": f"{self.insights['avg_load_time']:.2f}s",
            "top_issues": [f"{issue_type}: {count} pages" for issue_type, count in self.insights["top_issues"][:5]],
            "action_plan": self.get_action_plan()
        }
        
        # Create summary prompt
        summary_prompt = f"""
        Generate a concise SEO summary and action plan based on this data:
        
        {json.dumps(summary_data, indent=2)}
        
        Provide:
        1. A brief overall assessment (2-3 sentences)
        2. Top 3 strengths
        3. Top 3 areas for improvement
        4. A prioritized 3-step action plan
        
        Keep it concise and actionable.
        """
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant providing actionable advice."},
                {"role": "user", "content": summary_prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def _generate_summary_with_gemini(self):
        """Generate summary using Google's Gemini API"""
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Prepare summary data
        summary_data = {
            "website": self.analysis_results["url"],
            "overall_score": int(self.insights["overall_score"]),
            "grade": self.insights["grade"],
            "analyzed_pages": len(self.analysis_results["pages"]),
            "avg_word_count": int(self.insights["avg_word_count"]),
            "avg_load_time": f"{self.insights['avg_load_time']:.2f}s",
            "top_issues": [f"{issue_type}: {count} pages" for issue_type, count in self.insights["top_issues"][:5]],
            "action_plan": self.get_action_plan()
        }
        
        # Create summary prompt
        summary_prompt = f"""
        Generate a concise SEO summary and action plan based on this data:
        
        {json.dumps(summary_data, indent=2)}
        
        Provide:
        1. A brief overall assessment (2-3 sentences)
        2. Top 3 strengths
        3. Top 3 areas for improvement
        4. A prioritized 3-step action plan
        
        Keep it concise and actionable.
        """
        response = client.models.generate_content(
            model="gemini-pro",
            contents=summary_prompt,
        )
        
        return response.text
    
    def _generate_generic_summary(self):
        """Generate a generic summary based on analysis data"""
        score = int(self.insights["overall_score"])
        grade = self.insights["grade"]
        
        # Get score description
        if score >= 90:
            score_desc = "excellent"
        elif score >= 80:
            score_desc = "good"
        elif score >= 70:
            score_desc = "fair"
        elif score >= 60:
            score_desc = "needs improvement"
        else:
            score_desc = "poor"
        
        # Get top issues
        top_issues = [f"{issue_type}" for issue_type, count in self.insights["top_issues"][:3]]
        
        # Get action items
        action_plan = self.get_action_plan()
        action_items = [item["task"] for item in action_plan[:3]]
        
        # Build summary
        summary = f"""
        ## SEO Summary for {self.analysis_results["url"]}
        
        Overall SEO health is **{score_desc}** with a score of **{score}/100 (Grade {grade})**.
        
        Analysis of {len(self.analysis_results["pages"])} pages revealed an average word count of {int(self.insights["avg_word_count"])} words and average load time of {self.insights['avg_load_time']:.2f} seconds.
        
        ### Top Issues:
        {", ".join(top_issues)}
        
        ### Recommended Next Steps:
        1. {action_items[0] if len(action_items) > 0 else "No actions recommended"}
        {f"2. {action_items[1]}" if len(action_items) > 1 else ""}
        {f"3. {action_items[2]}" if len(action_items) > 2 else ""}
        """
        
        return summary

# Main Streamlit App
def main():
    # Load CSS and create navbar
    load_css()
    create_navbar()
    
    st.title("AI SEO Agent")
    st.write("Your autonomous SEO analysis and optimization assistant")
    
    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state['agent'] = None
    if 'analysis_complete' not in st.session_state:
        st.session_state['analysis_complete'] = False
    if 'current_tab' not in st.session_state:
        st.session_state['current_tab'] = "analysis"
    
    # API Keys in sidebar
    with st.sidebar:
        st.header("API Settings")
        
        openai_key = OPENAI_API_KEY
        google_key = GOOGLE_API_KEY
        gemini_key = GEMINI_API_KEY
        
        # Keep keys in session state
        if 'openai_key' not in st.session_state or st.session_state['openai_key'] != openai_key:
            st.session_state['openai_key'] = openai_key
        
        if 'google_key' not in st.session_state or st.session_state['google_key'] != google_key:
            st.session_state['google_key'] = google_key
            
        if 'gemini_key' not in st.session_state or st.session_state['gemini_key'] != gemini_key:
            st.session_state['gemini_key'] = gemini_key
        
        # Initialize agent if not already done
        if st.session_state['agent'] is None:
            st.session_state['agent'] = SEOAgent(
                openai_key=st.session_state.get('openai_key', ''),
                google_key=st.session_state.get('google_key', ''),
                gemini_key=st.session_state.get('gemini_key', '')
            )
        
        # Update agent keys if they've changed
        if st.session_state['agent'].openai_api_key != st.session_state.get('openai_key', ''):
            st.session_state['agent'].openai_api_key = st.session_state.get('openai_key', '')
            if st.session_state.get('openai_key', ''):
                st.session_state['agent'].openai_client = openai.OpenAI(api_key=st.session_state.get('openai_key', ''))
        
        if st.session_state['agent'].google_api_key != st.session_state.get('google_key', ''):
            st.session_state['agent'].google_api_key = st.session_state.get('google_key', '')
            
        if st.session_state['agent'].gemini_api_key != st.session_state.get('gemini_key', ''):
            st.session_state['agent'].gemini_api_key = st.session_state.get('gemini_key', '')
            if st.session_state.get('gemini_key', ''):
                genai.configure(api_key=st.session_state.get('gemini_key', ''))
        
        # Update agent capabilities
        st.session_state['agent'].capabilities = {
            "technical_seo": True,
            "content_analysis": bool(st.session_state.get('openai_key', '') or st.session_state.get('gemini_key', '')),
            "competitor_analysis": bool(st.session_state.get('google_key', '')),
            "keyword_research": bool(st.session_state.get('google_key', '') or st.session_state.get('openai_key', '') or st.session_state.get('gemini_key', ''))
        }
        
        # Display agent capabilities
        st.subheader("Agent Capabilities")
        st.markdown(f"✅ Technical SEO Analysis")
        st.markdown(f"{'✅' if st.session_state['agent'].capabilities['content_analysis'] else '❌'} Content Analysis")
        st.markdown(f"{'✅' if st.session_state['agent'].capabilities['competitor_analysis'] else '❌'} Competitor Analysis")
        st.markdown(f"{'✅' if st.session_state['agent'].capabilities['keyword_research'] else '❌'} Keyword Research")
        
        if not st.session_state['agent'].capabilities['content_analysis']:
            st.info("Add OpenAI or Gemini API key to enable content analysis")
        
        if not st.session_state['agent'].capabilities['competitor_analysis']:
            st.info("Add Google API key to enable competitor analysis")
    
    # Main content area
    if st.session_state['analysis_complete']:
        # Show tabs for different sections
        tab1, tab2, tab3, tab4 = st.tabs(["Analysis", "Recommendations", "Action Plan", "New Job"])
        
        # Set active tab based on session state
        if st.session_state['current_tab'] == "analysis":
            active_tab = tab1
        elif st.session_state['current_tab'] == "recommendations":
            active_tab = tab2
        elif st.session_state['current_tab'] == "action_plan":
            active_tab = tab3
        else:
            active_tab = tab4
        
        with tab1:
            st.session_state['current_tab'] = "analysis"
            display_analysis_results()
        
        with tab2:
            st.session_state['current_tab'] = "recommendations"
            display_recommendations()
        
        with tab3:
            st.session_state['current_tab'] = "action_plan"
            display_action_plan()
        
        with tab4:
            st.session_state['current_tab'] = "chat"
            # display_chat()
            display_analysis_form()
    else:
        # Show analysis form
        display_analysis_form()

# UI Component Functions
def display_analysis_form():
    """Display the analysis form"""
    st.subheader("Website Analysis")
    
    with st.form("seo_analysis_form"):
        url = st.text_input("Website URL", "https://www.linkedin.com")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            keywords = st.text_input("Target Keywords (comma separated)", "")
        with col2:
            page_limit = st.number_input("Page Limit", min_value=1, max_value=20, value=5)
        
        analysis_depth = st.select_slider(
            "Analysis Depth",
            options=["Basic", "Standard", "Deep"],
            value="Standard"
        )
        
        analyze_button = st.form_submit_button("Start Analysis", use_container_width=True)
        
        if analyze_button:
            if not url:
                st.error("Please enter a URL to analyze")
            else:
                # Run analysis in the background
                with st.spinner("Analyzing website..."):
                    results = st.session_state['agent'].analyze_website(
                        url=url,
                        keywords=keywords,
                        max_pages=page_limit,
                        analysis_depth=analysis_depth
                    )
                
                # Set flag to show results
                st.session_state['analysis_complete'] = True
                st.rerun()

def display_analysis_results():
    """Display the analysis results"""
    agent = st.session_state['agent']
    
    if not agent.analysis_results:
        st.error("No analysis data available. Please run an analysis first.")
        return
    
    # Display summary
    st.subheader("SEO Summary")
    seo_summary = agent.generate_seo_summary()
    st.markdown(seo_summary)
    
    # Display overall metrics
    st.subheader("Site Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        score = int(agent.insights["overall_score"])
        st.metric("Overall Score", f"{score}/100")
        st.markdown(f"""
        <div style='text-align:center; font-size:24px; font-weight:bold; 
                 color:{"#28a745" if score >= 80 else "#ffc107" if score >= 60 else "#dc3545"};'>
            Grade: {agent.insights["grade"]}
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_load = agent.insights["avg_load_time"]
        st.metric("Avg. Load Time", f"{avg_load:.2f}s")
        load_color = "#28a745" if avg_load < 2 else "#ffc107" if avg_load < 4 else "#dc3545"
        st.markdown(f"<div style='height:5px; background-color:{load_color};'></div>", unsafe_allow_html=True)
    
    with col3:
        avg_wc = int(agent.insights["avg_word_count"])
        st.metric("Avg. Word Count", f"{avg_wc}")
        wc_color = "#28a745" if avg_wc > 800 else "#ffc107" if avg_wc > 400 else "#dc3545"
        st.markdown(f"<div style='height:5px; background-color:{wc_color};'></div>", unsafe_allow_html=True)
    
    with col4:
        total_issues = sum(count for _, count in agent.insights["top_issues"])
        st.metric("Total Issues", total_issues)
    
    # Display page scores
    st.subheader("Page Scores")
    
    # Create DataFrame for pages
    df = pd.DataFrame([{
        'URL': p['url'],
        'Score': p['score'],
        'Title Length': p.get('title_length', 0),
        'Description Length': p.get('description_length', 0),
        'Word Count': p.get('word_count', 0),
        'Load Time (s)': round(p.get('load_time', 0), 2),
        'Issues': len(p.get('issues', []))
    } for p in agent.analysis_results["pages"]])
    
    st.dataframe(df, use_container_width=True)
    
    # Display keyword analysis if available
    if agent.insights["keyword_presence"]:
        st.subheader("Keyword Analysis")
        
        # Create table for keyword presence
        keyword_data = []
        
        for keyword, data in agent.insights["keyword_presence"].items():
            keyword_data.append({
                'Keyword': keyword,
                'In Title': f"{data['in_title_count']}/{len(agent.analysis_results['pages'])} ({data['in_title_percentage']:.0f}%)",
                'In Meta': f"{data['in_description_count']}/{len(agent.analysis_results['pages'])} ({data['in_description_percentage']:.0f}%)",
                'In Headings': f"{data['in_headings_count']}/{len(agent.analysis_results['pages'])} ({data['in_headings_percentage']:.0f}%)",
                'Avg. Density': f"{data['avg_density']:.2f}%"
            })
        
        keyword_df = pd.DataFrame(keyword_data)
        st.dataframe(keyword_df, use_container_width=True)
    
    # Display issues breakdown
    st.subheader("Issues Breakdown")
    
    # Create a bar chart of issues
    if agent.insights["top_issues"]:
        issue_types, issue_counts = zip(*agent.insights["top_issues"])
        
        issue_df = pd.DataFrame({
            'Issue Type': issue_types,
            'Count': issue_counts
        })
        
        st.bar_chart(issue_df.set_index('Issue Type'))
    else:
        st.info("No issues found.")
    
    # Display detailed page analysis in expandable sections
    st.subheader("Page Details")
    
    for page in agent.analysis_results["pages"]:
        with st.expander(f"{page['url']} - Score: {page['score']}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                # Score badge
                score_color = "#28a745" if page['score'] >= 80 else "#ffc107" if page['score'] >= 60 else "#dc3545"
                st.markdown(f"""
                <div style="padding:15px; background-color:{score_color}; 
                border-radius:5px; color:white; text-align:center;">
                    <h2 style="margin:0; color:white;">{page['score']}/100</h2>
                    <p style="margin:0; font-weight:bold;">Grade {agent._get_grade(page['score'])}</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Page metrics in columns
                mcol1, mcol2, mcol3 = st.columns(3)
                with mcol1:
                    st.metric("Word Count", page.get('word_count', 0))
                with mcol2:
                    st.metric("Load Time", f"{page.get('load_time', 0):.2f}s")
                with mcol3:
                    st.metric("Headings", f"H1: {page.get('h1_count', 0)}, H2: {page.get('h2_count', 0)}")
            
            # SEO elements
            st.write("#### SEO Elements")
            st.write(f"**Title:** {page.get('title', 'N/A')}")
            st.write(f"**Description:** {page.get('description', 'N/A')}")
            
            # Headings display
            if page.get('h1_tags') or page.get('h2_tags'):
                st.write("#### Heading Structure")
                
                if page.get('h1_tags'):
                    for h1 in page['h1_tags']:
                        st.markdown(f"<div style='padding:5px; background-color:#f0f0f0; border-left:3px solid #4267B2;'><strong>H1:</strong> {h1}</div>", unsafe_allow_html=True)
                
                if page.get('h2_tags'):
                    for h2 in page['h2_tags'][:3]:  # Show first 3 H2s
                        st.markdown(f"<div style='padding:5px; margin-left:20px; background-color:#f8f9fa; border-left:2px solid #6c757d;'><strong>H2:</strong> {h2}</div>", unsafe_allow_html=True)
                    
                    if len(page.get('h2_tags', [])) > 3:
                        st.markdown(f"<em>...and {len(page['h2_tags']) - 3} more H2 headings</em>", unsafe_allow_html=True)
            
            # Keyword presence if available
            if 'keyword_presence' in page:
                st.write("#### Keyword Presence")
                keyword_table = []
                
                for keyword, data in page['keyword_presence'].items():
                    keyword_table.append({
                        'Keyword': keyword,
                        'In Title': '✅' if data['in_title'] else '❌',
                        'In Description': '✅' if data['in_description'] else '❌',
                        'In Headings': '✅' if data['in_headings'] else '❌',
                        'Occurrences': data['occurrences'],
                        'Density': f"{data['density']:.2f}%"
                    })
                
                st.table(pd.DataFrame(keyword_table))
            
            # AI Content Analysis if available
            if 'ai_analysis' in page:
                st.write("#### AI Content Analysis")
                
                ai = page['ai_analysis']
                if "error" in ai:
                    st.error(f"Error in AI analysis: {ai['error']}")
                else:
                    ai_col1, ai_col2, ai_col3 = st.columns(3)
                    
                    with ai_col1:
                        content_score = ai.get('content_quality_score', 0)
                        score_color = "#28a745" if content_score >= 80 else "#ffc107" if content_score >= 60 else "#dc3545"
                        st.markdown(f"""
                        <div style="padding:10px; background-color:{score_color}; 
                        border-radius:5px; color:white; text-align:center;">
                            <h3 style="margin:0; color:white;">{content_score}</h3>
                            <p style="margin:0;">Content Quality</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ai_col2:
                        readability = ai.get('readability_score', 0)
                        score_color = "#28a745" if readability >= 80 else "#ffc107" if readability >= 60 else "#dc3545"
                        st.markdown(f"""
                        <div style="padding:10px; background-color:{score_color}; 
                        border-radius:5px; color:white; text-align:center;">
                            <h3 style="margin:0; color:white;">{readability}</h3>
                            <p style="margin:0;">Readability</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with ai_col3:
                        keyword_usage = ai.get('keyword_usage', 'N/A')
                        usage_color = {
                            'excellent': '#28a745',
                            'good': '#5cb85c',
                            'fair': '#ffc107',
                            'poor': '#dc3545'
                        }.get(keyword_usage.lower(), '#6c757d')
                        
                        st.markdown(f"""
                        <div style="padding:10px; background-color:{usage_color}; 
                        border-radius:5px; color:white; text-align:center;">
                            <h3 style="margin:0; color:white;">{keyword_usage.title()}</h3>
                            <p style="margin:0;">Keyword Usage</p>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Content strengths and weaknesses
                    if 'strengths' in ai and ai['strengths']:
                        st.write("**Content Strengths:**")
                        for strength in ai['strengths']:
                            st.markdown(f"<div class='tag tag-success'>✓</div> {strength}", unsafe_allow_html=True)
                    
                    if 'weaknesses' in ai and ai['weaknesses']:
                        st.write("**Content Weaknesses:**")
                        for weakness in ai['weaknesses']:
                            st.markdown(f"<div class='tag tag-danger'>!</div> {weakness}", unsafe_allow_html=True)
                    
                    if 'recommendations' in ai and ai['recommendations']:
                        st.write("**AI Recommendations:**")
                        for rec in ai['recommendations']:
                            st.markdown(f"<div class='tag'>→</div> {rec}", unsafe_allow_html=True)
            
            # Issues
            if page.get('issues'):
                st.write("#### Issues")
                for issue in page['issues']:
                    severity_class = f"priority-{issue['severity']}" if 'severity' in issue else ""
                    st.markdown(f"<div class='{severity_class}'><strong>{issue.get('message', '')}</strong></div>", unsafe_allow_html=True)

def display_recommendations():
    """Display the recommendations"""
    agent = st.session_state['agent']
    
    if not agent.recommendations:
        st.error("No recommendations available. Please run an analysis first.")
        return
    
    st.subheader("SEO Recommendations")
    
    # Critical recommendations
    if agent.recommendations.get("critical"):
        st.markdown("### Critical Issues")
        st.markdown("These issues require immediate attention and could be significantly impacting your SEO performance.")
        
        for rec in agent.recommendations["critical"]:
            with st.container():
                st.markdown(f"""
                <div class='analysis-container priority-high'>
                    <h4>{rec['message']}</h4>
                    <p>{rec['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show affected pages if available
                if 'affected_pages' in rec and rec['affected_pages']:
                    with st.expander("Affected Pages"):
                        for page in rec['affected_pages'][:10]:  # Show first 10 pages
                            st.markdown(f"- {page}")
                        
                        if len(rec['affected_pages']) > 10:
                            st.markdown(f"...and {len(rec['affected_pages']) - 10} more pages")
    
    # Important recommendations
    if agent.recommendations.get("important"):
        st.markdown("### Important Improvements")
        st.markdown("These issues should be addressed to improve your SEO performance.")
        
        for rec in agent.recommendations["important"]:
            with st.container():
                st.markdown(f"""
                <div class='analysis-container priority-medium'>
                    <h4>{rec['message']}</h4>
                    <p>{rec['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show affected pages if available
                if 'affected_pages' in rec and rec['affected_pages']:
                    with st.expander("Affected Pages"):
                        for page in rec['affected_pages'][:10]:  # Show first 10 pages
                            st.markdown(f"- {page}")
                        
                        if len(rec['affected_pages']) > 10:
                            st.markdown(f"...and {len(rec['affected_pages']) - 10} more pages")
    
    # Opportunities
    if agent.recommendations.get("opportunity"):
        st.markdown("### Opportunities")
        st.markdown("These are opportunities to further enhance your SEO performance.")
        
        for rec in agent.recommendations["opportunity"]:
            with st.container():
                st.markdown(f"""
                <div class='analysis-container priority-low'>
                    <h4>{rec['message']}</h4>
                    <p>{rec['details']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Show affected pages if available
                if 'affected_pages' in rec and rec['affected_pages']:
                    with st.expander("Affected Pages"):
                        for page in rec['affected_pages'][:10]:  # Show first 10 pages
                            st.markdown(f"- {page}")
                        
                        if len(rec['affected_pages']) > 10:
                            st.markdown(f"...and {len(rec['affected_pages']) - 10} more pages")
                
                # Show competitors if available
                if 'competitors' in rec:
                    with st.expander("Competitors to Analyze"):
                        for competitor in rec['competitors']:
                            st.markdown(f"- {competitor}")

def display_action_plan():
    """Display the action plan"""
    agent = st.session_state['agent']
    
    if not agent.recommendations:
        st.error("No action plan available. Please run an analysis first.")
        return
    
    st.subheader("SEO Action Plan")
    st.markdown("Here's a prioritized action plan to improve your SEO performance:")
    
    action_plan = agent.get_action_plan()
    
    if not action_plan:
        st.info("No actions recommended at this time.")
        return
    
    # Display tasks by priority
    high_priority = [task for task in action_plan if task['priority'] == 'high']
    medium_priority = [task for task in action_plan if task['priority'] == 'medium']
    low_priority = [task for task in action_plan if task['priority'] == 'low']
    
    # High priority tasks
    if high_priority:
        st.markdown("### Immediate Actions (1-2 Weeks)")
        
        for i, task in enumerate(high_priority):
            st.write(f"**Step {i+1}:** {task['task']}")
            st.write(f"   - {task['details']}")
    
    # Medium priority tasks
    if medium_priority:
        st.markdown("### Short-Term Actions (1-2 Months)")
        
        for i, task in enumerate(medium_priority):
            
            st.write(f"**Step {i+1}:** {task['task']}")
            st.write(f"   - {task['details']}")
    
    # Low priority tasks
    if low_priority:
        st.markdown("### Long-Term Actions (2+ Months)")
        
        for i, task in enumerate(low_priority):
            
            st.write(f"**Step {i+1}:** {task['task']}")
            st.write(f"   - {task['details']}")
            
    
    # Download action plan button
    action_plan_text = "# SEO Action Plan\n\n"
    
    if high_priority:
        action_plan_text += "## Immediate Actions (1-2 Weeks)\n\n"
        for i, task in enumerate(high_priority):
            action_plan_text += f"{i+1}. **{task['task']}**\n   - {task['details']}\n\n"
    
    if medium_priority:
        action_plan_text += "## Short-Term Actions (1-2 Months)\n\n"
        for i, task in enumerate(medium_priority):
            action_plan_text += f"{i+1}. **{task['task']}**\n   - {task['details']}\n\n"
    
    if low_priority:
        action_plan_text += "## Long-Term Actions (2+ Months)\n\n"
        for i, task in enumerate(low_priority):
            action_plan_text += f"{i+1}. **{task['task']}**\n   - {task['details']}\n\n"
    
    st.download_button(
        "Download Action Plan",
        action_plan_text,
        "seo_action_plan.md",
        "text/markdown",
        key='download-action-plan'
    )

def display_chat():
    """Display the chat interface with the SEO agent"""
    agent = st.session_state['agent']
    
    st.subheader("Chat with SEO Agent")
    st.markdown("Ask questions about your analysis or get specific recommendations")
    
    # Initialize chat input
    if 'chat_input' not in st.session_state:
        st.session_state['chat_input'] = ""
    
    # Display chat history
    st.write("### Chat History")
    
    for message in agent.chat_history:
        if message['role'] == 'user':
            st.write(f"**You:** {message['content']}")
        else:
            st.write(f"**Agent:** {message['content']}")
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Chat input
    with st.form("chat_form", clear_on_submit=True):
        st.session_state['chat_input'] = st.text_input("Your question:", key="chat_question")
        submit_button = st.form_submit_button("Send")
        
        if submit_button and st.session_state['chat_input']:
            user_message = st.session_state['chat_input']
            
            # Add user message to chat history
            agent.add_message("user", user_message)
            
            # Process the message and generate a response
            if not agent.analysis_results:
                agent.add_message("agent", "I don't have any analysis data yet. Please run an analysis first.")
            else:
                # Generate a response based on the user's question
                if agent.openai_api_key:
                    response = process_question_with_openai(agent, user_message)
                elif agent.gemini_api_key:
                    response = process_question_with_gemini(agent, user_message)
                else:
                    response = "I'm currently in limited functionality mode because no AI API keys are provided. For best results, please add an OpenAI or Gemini API key in the sidebar."
                
                agent.add_message("agent", response)
            
            # Clear the input and rerun to update the display
            st.session_state['chat_input'] = ""
            st.rerun()

def process_question_with_openai(agent, question):
    """Process a user question using OpenAI API"""
    try:
        # Prepare context for the AI
        insights_summary = json.dumps(agent.insights, indent=2)
        recommendations_summary = json.dumps(agent.recommendations, indent=2)
        
        prompt = f"""
        You are an SEO expert assistant. You have analyzed a website and have the following data:
        
        INSIGHTS: {insights_summary}
        
        RECOMMENDATIONS: {recommendations_summary}
        
        Based on this data, please answer the following question from the user:
        
        USER QUESTION: {question}
        
        Provide a helpful, concise response based only on the data provided. If the question cannot be answered with the available data, explain what information would be needed.
        """
        
        client = openai.OpenAI(api_key=agent.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant providing actionable advice based on website analysis data."},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

def process_question_with_gemini(agent, question):
    """Process a user question using Gemini API"""
    try:
        # Configure the API client
        genai.configure(api_key=GEMINI_API_KEY)
        
        # Create model instance
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Prepare context for the AI
        insights_summary = json.dumps(agent.insights, indent=2)
        recommendations_summary = json.dumps(agent.recommendations, indent=2)
        
        prompt = f"""
        You are an SEO expert assistant. You have analyzed a website and have the following data:
        
        INSIGHTS: {insights_summary}
        
        RECOMMENDATIONS: {recommendations_summary}
        
        Based on this data, please answer the following question from the user:
        
        USER QUESTION: {question}
        
        Provide a helpful, concise response based only on the data provided. If the question cannot be answered with the available data, explain what information would be needed.
        """
        
        # Generate content using the model
        response = model.generate_content(prompt)
        
        return response.text
        
    except Exception as e:
        return f"Sorry, I encountered an error while processing your question: {str(e)}"

if __name__ == '__main__':
    password = st.text_input("Enter password to access the app:", type="password")
    if password == "Napster":
        st.session_state['password'] = password
        main()
    else:
        st.warning("Please enter the correct password to access the app.")
    
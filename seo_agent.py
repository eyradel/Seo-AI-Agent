from urllib.parse import urlparse
import re
import time
import json
from datetime import datetime
import os
import requests
from bs4 import BeautifulSoup
from googleapiclient.discovery import build
from google import genai
import openai


OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class SEOAgent:
    def __init__(self, openai_key=None, google_key=None, gemini_key=None):
        self.openai_api_key = openai_key or OPENAI_API_KEY
        self.google_api_key = google_key or GOOGLE_API_KEY
        self.gemini_api_key = gemini_key or GEMINI_API_KEY
        self.chat_history = []
        self.analysis_results = {}
        self.recommendations = {}
        self.insights = {}
        self.competitor_data = {}
        self.status = "idle"

        if self.openai_api_key:
            self.openai_client = openai.OpenAI(api_key=self.openai_api_key)

        self.capabilities = {
            "technical_seo": True,
            "content_analysis": bool(self.openai_api_key),
            "competitor_analysis": bool(self.google_api_key),
            "keyword_research": bool(self.google_api_key or self.openai_api_key or self.gemini_api_key),
        }

    def add_message(self, role, content):
        self.chat_history.append({"role": role, "content": content, "timestamp": datetime.now()})
        return len(self.chat_history) - 1

    def analyze_website(self, url, keywords=None, max_pages=5, analysis_depth="Standard"):
        self.status = "analyzing"
        self.add_message("agent", f"Starting analysis of {url}. This may take a few minutes...")

        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        base_domain = urlparse(url).netloc
        results = self._crawl_site(url, max_pages, base_domain)

        keyword_list = []
        if keywords:
            keyword_list = [k.strip() for k in keywords.split(",")]
            self.add_message("agent", f"Analyzing keyword presence for: {', '.join(keyword_list)}")

        if self.capabilities["content_analysis"] and analysis_depth != "Basic":
            self._enhance_with_ai(results, keyword_list, analysis_depth)

        if self.capabilities["competitor_analysis"] and keyword_list and analysis_depth == "Deep":
            self._analyze_competitors(base_domain, keyword_list)

        self._generate_insights(results, keyword_list, base_domain)

        self.analysis_results = {
            "url": url,
            "date": datetime.now().strftime("%Y-%m-%d %H:%M"),
            "pages": results,
            "keywords": keyword_list,
            "domain": base_domain,
        }

        self.status = "complete"
        self.add_message("agent", "Analysis complete! I've generated insights and recommendations for your website.")

        return results

    def _crawl_site(self, start_url, max_pages, base_domain):
        results = []
        to_crawl = {start_url}
        crawled = set()

        self.add_message("agent", f"Crawling website starting from {start_url}...")

        while to_crawl and len(crawled) < max_pages:
            current_url = to_crawl.pop()
            if current_url in crawled:
                continue

            self.add_message("agent", f"Analyzing page: {current_url}")

            try:
                start_time = time.time()
                response = requests.get(current_url, timeout=15)
                load_time = time.time() - start_time

                if response.status_code == 200:
                    html = response.text
                    page_data = self._extract_page_data(html, current_url, load_time)
                    internal_links = self._extract_links(html, current_url, base_domain)
                    for link in internal_links:
                        if link not in crawled and len(to_crawl) + len(crawled) < max_pages:
                            to_crawl.add(link)
                    results.append(page_data)
                crawled.add(current_url)
            except Exception as e:
                self.add_message("agent", f"Error analyzing {current_url}: {str(e)}")
                crawled.add(current_url)

        self.add_message("agent", f"Finished crawling. Analyzed {len(results)} pages.")
        return results

    def _extract_page_data(self, html, url, load_time):
        soup = BeautifulSoup(html, "html.parser")
        title = soup.title.text if soup.title else "No title found"
        meta_desc_tag = soup.find("meta", attrs={"name": "description"})
        description = meta_desc_tag.get("content") if meta_desc_tag else "No description found"
        for script in soup(["script", "style"]):
            script.extract()
        text_content = soup.get_text(separator=" ")
        text_content = re.sub(r"\s+", " ", text_content).strip()
        text_sample = text_content[:5000] if len(text_content) > 5000 else text_content
        word_count = len(re.findall(r"\b\w+\b", text_content))
        h1_tags = [h1.text.strip() for h1 in soup.find_all("h1")]
        h2_tags = [h2.text.strip() for h2 in soup.find_all("h2")]
        h3_tags = [h3.text.strip() for h3 in soup.find_all("h3")]
        img_tags = soup.find_all("img")
        img_count = len(img_tags)
        img_missing_alt = sum(1 for img in img_tags if not img.get("alt"))
        schema_tags = soup.find_all("script", type="application/ld+json")
        has_schema = len(schema_tags) > 0
        viewport_tag = soup.find("meta", attrs={"name": "viewport"})
        has_viewport = bool(viewport_tag)
        page_size = len(html) / 1024

        score, issues = self._calculate_seo_score({
            "title": title,
            "title_length": len(title),
            "description": description,
            "description_length": len(description) if description != "No description found" else 0,
            "word_count": word_count,
            "h1_count": len(h1_tags),
            "img_count": img_count,
            "img_missing_alt": img_missing_alt,
            "load_time": load_time,
            "has_schema": has_schema,
            "has_viewport": has_viewport,
            "page_size": page_size,
        })

        return {
            "url": url,
            "title": title,
            "title_length": len(title),
            "description": description,
            "description_length": len(description) if description != "No description found" else 0,
            "text_sample": text_sample,
            "word_count": word_count,
            "h1_tags": h1_tags,
            "h2_tags": h2_tags,
            "h3_tags": h3_tags,
            "h1_count": len(h1_tags),
            "h2_count": len(h2_tags),
            "h3_count": len(h3_tags),
            "img_count": img_count,
            "img_missing_alt": img_missing_alt,
            "has_schema": has_schema,
            "has_viewport": has_viewport,
            "load_time": load_time,
            "page_size": page_size,
            "score": score,
            "issues": issues,
        }

    def _extract_links(self, html, current_url, base_domain):
        soup = BeautifulSoup(html, "html.parser")
        internal_links = []
        for a_tag in soup.find_all("a", href=True):
            link = a_tag["href"]
            if link.startswith("#") or link.startswith("javascript:"):
                continue
            if link.startswith("/"):
                full_url = f"{urlparse(current_url).scheme}://{base_domain}{link}"
                internal_links.append(full_url)
            elif base_domain in link:
                internal_links.append(link)
        return internal_links

    def _calculate_seo_score(self, page_data):
        score = 100
        issues = []
        if page_data["title"] == "No title found":
            score -= 20
            issues.append({"type": "title", "severity": "high", "message": "Missing title tag"})
        elif page_data["title_length"] < 30:
            score -= 10
            issues.append({"type": "title", "severity": "medium", "message": "Title too short (under 30 characters)"})
        elif page_data["title_length"] > 60:
            score -= 5
            issues.append({"type": "title", "severity": "low", "message": "Title too long (over 60 characters)"})

        if page_data["description"] == "No description found":
            score -= 15
            issues.append({"type": "meta", "severity": "high", "message": "Missing meta description"})
        elif page_data["description_length"] < 80:
            score -= 10
            issues.append({"type": "meta", "severity": "medium", "message": "Meta description too short (under 80 characters)"})
        elif page_data["description_length"] > 160:
            score -= 5
            issues.append({"type": "meta", "severity": "low", "message": "Meta description too long (over 160 characters)"})

        if page_data["word_count"] < 300:
            score -= 15
            issues.append({"type": "content", "severity": "high", "message": "Low word count (under 300 words)"})
        elif page_data["word_count"] < 600:
            score -= 5
            issues.append({"type": "content", "severity": "low", "message": "Content could be more comprehensive (under 600 words)"})

        if page_data["h1_count"] == 0:
            score -= 10
            issues.append({"type": "headings", "severity": "high", "message": "Missing H1 heading"})

        if page_data["img_count"] > 0 and page_data["img_missing_alt"] > 0:
            score -= min(10, page_data["img_missing_alt"] * 2)
            percent_missing = (page_data["img_missing_alt"] / page_data["img_count"]) * 100 if page_data["img_count"] else 0
            severity = "high" if percent_missing > 50 else "medium"
            issues.append({"type": "images", "severity": severity, "message": f"{page_data['img_missing_alt']} images missing alt text"})

        if page_data["load_time"] > 5:
            score -= 15
            issues.append({"type": "performance", "severity": "high", "message": f"Very slow load time ({page_data['load_time']:.2f}s)"})
        elif page_data["load_time"] > 3:
            score -= 10
            issues.append({"type": "performance", "severity": "medium", "message": f"Slow load time ({page_data['load_time']:.2f}s)"})

        if not page_data["has_schema"]:
            score -= 5
            issues.append({"type": "schema", "severity": "medium", "message": "No structured data (Schema.org) found"})

        if not page_data["has_viewport"]:
            score -= 10
            issues.append({"type": "mobile", "severity": "high", "message": "No viewport meta tag (not mobile-friendly)"})

        if page_data["page_size"] > 5000:
            score -= 10
            issues.append({"type": "performance", "severity": "high", "message": f"Page size too large ({page_data['page_size']/1024:.1f} MB)"})
        elif page_data["page_size"] > 2000:
            score -= 5
            issues.append({"type": "performance", "severity": "medium", "message": f"Page size large ({page_data['page_size']/1024:.1f} MB)"})

        score = max(0, min(100, score))
        return score, issues

    def _enhance_with_ai(self, results, keyword_list, depth):
        if not self.openai_api_key:
            self.add_message("agent", "AI content analysis unavailable. Add an OpenAI API key.")
            return
        self.add_message("agent", "Performing AI content analysis...")
        for result in results:
            result["ai_analysis"] = self._analyze_with_openai(result["title"], result["text_sample"], keyword_list)
            if keyword_list:
                result["keyword_presence"] = self._analyze_keyword_presence(
                    result["title"],
                    result["description"],
                    result["h1_tags"] + result["h2_tags"],
                    result["text_sample"],
                    keyword_list,
                )

    def _analyze_with_openai(self, title, content, keywords=None):
        try:
            prompt = f"""
            Analyze this webpage content from an SEO perspective:

            TITLE: {title}

            CONTENT SAMPLE: {content[:1500]}...
            """
            if keywords:
                prompt += f"\nTARGET KEYWORDS: {', '.join(keywords)}\n\n"
            prompt += """
            Please provide the following analysis in JSON format:
            1. content_quality_score
            2. readability_score
            3. keyword_usage
            4. strengths
            5. weaknesses
            6. recommendations
            Return only valid JSON with these exact keys.
            """
            response = self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "You are an expert SEO consultant analyzing web content."},
                    {"role": "user", "content": prompt},
                ],
            )
            ai_analysis = json.loads(response.choices[0].message.content)
            return ai_analysis
        except Exception as e:
            self.add_message("agent", f"Error in OpenAI analysis: {str(e)}")
            return {"error": str(e)}

    def _analyze_keyword_presence(self, title, description, headings, content, keywords):
        keyword_presence = {}
        for keyword in keywords:
            keyword_lower = keyword.lower()
            in_title = keyword_lower in title.lower()
            in_description = description != "No description found" and keyword_lower in description.lower()
            in_headings = any(keyword_lower in h.lower() for h in headings)
            occurrences = content.lower().count(keyword_lower)
            word_count = len(re.findall(r"\b\w+\b", content))
            density = (occurrences / word_count * 100) if word_count > 0 else 0
            keyword_presence[keyword] = {
                "in_title": in_title,
                "in_description": in_description,
                "in_headings": in_headings,
                "occurrences": occurrences,
                "density": density,
            }
        return keyword_presence

    def _analyze_competitors(self, domain, keywords):
        if not self.google_api_key:
            self.add_message("agent", "Competitor analysis unavailable. No Google API key provided.")
            return
        self.add_message("agent", "Analyzing search competition...")
        serp_data = {}
        for keyword in keywords:
            self.add_message("agent", f"Checking SERP for: {keyword}")
            serp_data[keyword] = self._get_serp_data(keyword, domain)
        self.competitor_data = serp_data

    def _get_serp_data(self, keyword, domain):
        try:
            service = build("customsearch", "v1", developerKey=self.google_api_key)
            result = service.cse().list(q=keyword, cx="YOUR_CUSTOM_SEARCH_ENGINE_ID", num=10).execute()
            serp_data = {"total_results": result.get("searchInformation", {}).get("totalResults", 0), "domain_position": None, "top_results": []}
            if "items" in result:
                for i, item in enumerate(result["items"]):
                    result_domain = urlparse(item["link"]).netloc
                    if domain in result_domain:
                        serp_data["domain_position"] = i + 1
                    serp_data["top_results"].append({
                        "position": i + 1,
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "domain": result_domain,
                        "snippet": item.get("snippet", ""),
                        "is_our_domain": domain in result_domain,
                    })
            return serp_data
        except Exception as e:
            self.add_message("agent", f"Error retrieving SERP data: {str(e)}")
            return {"error": str(e)}

    def _generate_insights(self, results, keywords, domain):
        self.add_message("agent", "Generating insights and recommendations...")
        avg_score = sum(page["score"] for page in results) / len(results) if results else 0
        avg_word_count = sum(page["word_count"] for page in results) / len(results) if results else 0
        avg_load_time = sum(page["load_time"] for page in results) / len(results) if results else 0
        all_issues = []
        for page in results:
            for issue in page["issues"]:
                all_issues.append({"url": page["url"], "issue": issue})
        issue_counts = {}
        for issue in all_issues:
            issue_type = issue["issue"]["type"]
            issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        self.insights = {
            "overall_score": avg_score,
            "grade": self._get_grade(avg_score),
            "avg_word_count": avg_word_count,
            "avg_load_time": avg_load_time,
            "top_issues": sorted(issue_counts.items(), key=lambda x: x[1], reverse=True),
            "keyword_presence": self._analyze_site_keyword_presence(results, keywords) if keywords else None,
        }
        self._generate_recommendations(results, keywords, domain)

    def _analyze_site_keyword_presence(self, results, keywords):
        site_keyword_data = {}
        for keyword in keywords:
            present_in_title = sum(1 for r in results if "keyword_presence" in r and keyword in r["keyword_presence"] and r["keyword_presence"][keyword]["in_title"])
            present_in_desc = sum(1 for r in results if "keyword_presence" in r and keyword in r["keyword_presence"] and r["keyword_presence"][keyword]["in_description"])
            present_in_h = sum(1 for r in results if "keyword_presence" in r and keyword in r["keyword_presence"] and r["keyword_presence"][keyword]["in_headings"])
            total_occurrences = sum(r["keyword_presence"][keyword]["occurrences"] for r in results if "keyword_presence" in r and keyword in r["keyword_presence"])
            avg_density = sum(r["keyword_presence"][keyword]["density"] for r in results if "keyword_presence" in r and keyword in r["keyword_presence"]) / len(results) if results else 0
            site_keyword_data[keyword] = {
                "in_title_count": present_in_title,
                "in_title_percentage": (present_in_title / len(results) * 100) if results else 0,
                "in_description_count": present_in_desc,
                "in_description_percentage": (present_in_desc / len(results) * 100) if results else 0,
                "in_headings_count": present_in_h,
                "in_headings_percentage": (present_in_h / len(results) * 100) if results else 0,
                "total_occurrences": total_occurrences,
                "avg_density": avg_density,
            }
        return site_keyword_data

    def _generate_recommendations(self, results, keywords, domain):
        recommendations = {"critical": [], "important": [], "opportunity": []}
        missing_titles = [p["url"] for p in results if p.get("title") == "No title found"]
        if missing_titles:
            recommendations["critical"].append({
                "type": "title",
                "message": f"Add title tags to {len(missing_titles)} pages",
                "details": "Title tags are critical for SEO and user experience",
                "affected_pages": missing_titles,
            })
        missing_desc = [p["url"] for p in results if p.get("description") == "No description found"]
        if missing_desc:
            recommendations["important"].append({
                "type": "meta",
                "message": f"Add meta descriptions to {len(missing_desc)} pages",
                "details": "Meta descriptions improve click-through rates from search results",
                "affected_pages": missing_desc,
            })
        missing_h1 = [p["url"] for p in results if p.get("h1_count", 0) == 0]
        if missing_h1:
            recommendations["important"].append({
                "type": "headings",
                "message": f"Add H1 headings to {len(missing_h1)} pages",
                "details": "Every page should have a single H1 heading that clearly describes the content",
                "affected_pages": missing_h1,
            })
        pages_with_missing_alt = [p for p in results if p.get("img_count", 0) > 0 and p.get("img_missing_alt", 0) > 0]
        if pages_with_missing_alt:
            total_missing = sum(p.get("img_missing_alt", 0) for p in pages_with_missing_alt)
            recommendations["important"].append({
                "type": "images",
                "message": f"Add alt text to {total_missing} images across {len(pages_with_missing_alt)} pages",
                "details": "Alt text improves accessibility and helps search engines understand images",
                "affected_pages": [p["url"] for p in pages_with_missing_alt],
            })
        self.recommendations = recommendations

    def _get_grade(self, score):
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
        if not self.recommendations:
            return []
        action_plan = []
        for rec in self.recommendations.get("critical", []):
            action_plan.append({"priority": "high", "task": rec["message"], "details": rec["details"], "type": rec["type"]})
        for rec in self.recommendations.get("important", []):
            action_plan.append({"priority": "medium", "task": rec["message"], "details": rec["details"], "type": rec["type"]})
        for rec in self.recommendations.get("opportunity", [])[:3]:
            action_plan.append({"priority": "low", "task": rec["message"], "details": rec["details"], "type": rec["type"]})
        return action_plan

    def generate_seo_summary(self):
        if not self.analysis_results or not self.insights:
            return "No analysis data available. Run analysis first."
        try:
            if self.openai_api_key:
                return self._generate_summary_with_openai()
            else:
                return self._generate_generic_summary()
        except Exception:
            return self._generate_generic_summary()

    def _generate_summary_with_openai(self):
        client = openai.OpenAI(api_key=self.openai_api_key)
        summary_data = {
            "website": self.analysis_results["url"],
            "overall_score": int(self.insights.get("overall_score", 0)),
            "grade": self.insights.get("grade", ""),
            "analyzed_pages": len(self.analysis_results.get("pages", [])),
            "avg_word_count": int(self.insights.get("avg_word_count", 0)),
            "avg_load_time": f"{self.insights.get('avg_load_time', 0):.2f}s",
            "top_issues": [f"{issue_type}: {count} pages" for issue_type, count in self.insights.get("top_issues", [])[:5]],
            "action_plan": self.get_action_plan(),
        }
        summary_prompt = f"""
        Generate a concise SEO summary and action plan based on this data:

        {json.dumps(summary_data, indent=2)}

        Provide: 1) brief assessment, 2) top strengths, 3) top improvements, 4) 3-step plan.
        Keep it concise and actionable.
        """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert SEO consultant."},
                {"role": "user", "content": summary_prompt},
            ],
        )
        return response.choices[0].message.content

    def _generate_generic_summary(self):
        score = int(self.insights.get("overall_score", 0))
        grade = self.insights.get("grade", "")
        top_issues = [f"{issue_type}" for issue_type, _ in self.insights.get("top_issues", [])[:3]]
        action_items = [item["task"] for item in self.get_action_plan()[:3]]
        summary = f"""
        ## SEO Summary for {self.analysis_results.get('url','')}

        Overall SEO health is {grade} with a score of {score}/100.

        Top issues: {', '.join(top_issues)}

        Next steps:
        1. {action_items[0] if len(action_items) > 0 else 'No actions recommended'}
        {f'2. {action_items[1]}' if len(action_items) > 1 else ''}
        {f'3. {action_items[2]}' if len(action_items) > 2 else ''}
        """
        return summary



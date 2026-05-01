import re
import json
import os

def parse_readme(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split into sections based on ## headers
    sections = re.split(r'\n## ', content)
    
    data = {
        "categories": [],
        "trending": [],
        "roadmap": "",
        "quick_start": []
    }

    for section in sections:
        lines = section.split('\n')
        title = lines[0].strip()
        
        # Skip Intro and TOC
        if any(x in title.lower() for x in ["ultimate-ai-resources", "why this list", "contents", "contribute"]):
            continue

        # Handle Roadmap (Mermaid)
        if "roadmap" in title.lower():
            mermaid_match = re.search(r'```mermaid\n(.*?)\n```', section, re.DOTALL)
            if mermaid_match:
                data["roadmap"] = mermaid_match.group(1).strip()
            continue

        # Handle Quick Start
        if "quick start" in title.lower():
            options = re.split(r'#### ', section)
            for opt in options[1:]:
                opt_lines = opt.split('\n')
                opt_title = opt_lines[0].strip()
                code_match = re.search(r'```python\n(.*?)\n```', opt, re.DOTALL)
                if code_match:
                    data["quick_start"].append({
                        "title": opt_title,
                        "code": code_match.group(1).strip()
                    })
            continue

        # Handle Trending
        if "trending" in title.lower():
            trending_subsections = re.split(r'\n### ', section)
            for sub in trending_subsections[1:]:
                sub_lines = sub.split('\n')
                sub_title = sub_lines[0].strip()
                items = re.findall(r'- \*\*\[(.*?)\]\((.*?)\)\*\* – (.*)', sub)
                for name, link, desc in items:
                    data["trending"].append({
                        "name": name,
                        "link": link,
                        "description": desc,
                        "tag": sub_title
                    })
            continue

        # Standard Categories
        category = {
            "name": title,
            "subcategories": []
        }
        
        subsections = re.split(r'\n### ', section)
        for sub in subsections:
            sub_lines = sub.split('\n')
            if not sub_lines: continue
            
            sub_title = sub_lines[0].strip()
            if sub_title == title: continue # It's the main section intro
            
            subcategory = {
                "name": sub_title,
                "items": []
            }
            
            # Match - [Name](Link) – Description. or - [Name](Link)
            items = re.findall(r'- \[(.*?)\]\((.*?)\)(?: – (.*))?', sub)
            for name, link, desc in items:
                # Determine difficulty or type based on subcategory name or content
                difficulty = "Beginner"
                if any(x in sub_title.lower() for x in ["expert", "advanced", "production", "monitoring", "deployment"]):
                    difficulty = "Advanced"
                elif any(x in sub_title.lower() for x in ["intermediate", "frameworks", "libraries", "deep learning"]):
                    difficulty = "Intermediate"
                
                subcategory["items"].append({
                    "name": name,
                    "link": link,
                    "description": desc.strip() if desc else "",
                    "difficulty": difficulty
                })
            
            if subcategory["items"]:
                category["subcategories"].append(subcategory)
        
        if category["subcategories"]:
            data["categories"].append(category)

    return data

if __name__ == "__main__":
    readme_path = os.path.join(os.path.dirname(__file__), '../README.md')
    output_path = os.path.join(os.path.dirname(__file__), '../src/data/resources.json')
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    parsed_data = parse_readme(readme_path)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(parsed_data, f, indent=2)
    
    print(f"Successfully parsed README.md to {output_path}")

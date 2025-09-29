import re
import json

def extract_email(text):
    match = re.search(r"\S+@\S+", text)
    return match.group(0) if match else None

def extract_phone(text):
    match = re.search(r"\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}", text)
    return match.group(0) if match else None

def extract_city_state(text):
    match = re.search(r"([A-Za-z\s]+),\s*([A-Z]{2})", text)
    return (match.group(1).strip(), match.group(2)) if match else (None, None)

def extract_years(text):
    return re.findall(r"(?:19|20)\d{2}|current|Current", text)

# Personal Info 
def parse_personal_info(text):
    email = extract_email(text)
    phone = extract_phone(text)
    city, state = extract_city_state(text)
    return {"Email": email, "Phone": phone, "City": city, "State": state}

# Education 
def parse_education(text):
    years = extract_years(text)
    start, end = None, None
    if len(years) >= 2:
        start, end = years[0], years[1]

    city, state = None, None
    if end:
        # Take substring after the end year → likely contains city/state
        after_years = text.split(end, 1)[-1]
        match = re.search(r"([A-Za-z\s]+)[,;]?\s*([A-Z]{2})", after_years)
        if match:
            city, state = match.group(1).strip(), match.group(2).strip()

    # Degree = text before the first year
    degree = text.split(start)[0].strip() if start else text

    return {
        "Degree": degree,
        "Graduation_Start_Date": start,
        "Graduation_End_Date": end,
        "City": city,
        "State": state
    }

# Work Experience
def parse_work_experience(text, max_title_words=6):
    experiences = []

    DATE_PATTERN = re.compile(
        r"((?:Jan|Feb|Mar|Apr|May|Jun|July?|Aug|Sep|Sept|Oct|Nov|Dec)\s+\d{4})\s*[-–]?\s*((?:current|Current|\d{4}))"
        r"|(\d{4})\s*[-–]?\s*((?:current|Current|\d{4}))",
        flags=re.IGNORECASE
    )

    matches = list(DATE_PATTERN.finditer(text))

    for i, match in enumerate(matches):
        # Normalize start/end
        groups = [g for g in match.groups() if g]
        start_date = groups[0] if len(groups) > 0 else None
        end_date = groups[1] if len(groups) > 1 else None

        # Block for this job until next match or end of text
        next_start = matches[i+1].start() if i+1 < len(matches) else len(text)
        block = text[match.start():next_start]

        # Job Title = before the date
        pre_text = text[max(0, match.start()-80):match.start()].strip()
        pre_words = pre_text.split()
        job_title = " ".join(pre_words[-max_title_words:]) if pre_words else None

        # City/State (first after date)
        cs_match = re.search(r"([A-Za-z .'-]+?)\s*[,;]?\s+([A-Z]{2})\b", block)
        if cs_match:
            city, state = cs_match.group(1).strip(), cs_match.group(2).strip()
            description = block[cs_match.end():].strip()
        else:
            city, state, description = None, None, block.strip()

        experiences.append({
            "Job_Title": job_title,
            "Start_Date": start_date,
            "End_Date": end_date,
            "City": city,
            "State": state,
            "Description": description
        })

    return experiences

# Skills
def parse_skills(text: str):
    """Parse skills string into a list (split by spaces)."""
    if not text:
        return []
    return text.split()

# Main Parser 
def parse_resume(resume_json):
    page = resume_json["page_1"]
    return {
        "Name": page.get("name"),
        "Personal_Info": parse_personal_info(page.get("personal information", "")),
        "Education": parse_education(page.get("education", "")),
        "Work_Experience": parse_work_experience(page.get("work experience", "")),
        "Skills": parse_skills(page.get("skills", ""))
    }


# Example Run 
data = {
       "page_1": {
        "name": "NEHEMIAH BISHOP",
        "skills": "Jira Power Bl AgileCraft Microsoft Azure AB Testing Kanban Boards Logility Productboard Optimizely",
        "work experience": "AI Product Manager The Hershey Company 2019 current Hershey, PA Launched Logility demand forecasting models, leading to a 17% reduction in excess inventory Utilized Productboard Al to analyze and act on customer feedback, which resulted in 33% YoY revenue growth Grew market share by 19% through AgileCraft to scale chocolate products that increased customer satisfaction and repeat purchases Generated actionable customer trends insights using Power Bl, which resulted in a 47% increase in average time spent on chocolate product pages Product Manager UPMC 2016 2019 Pittsburgh, PA Led the launch of 6 new medical devices that improved patient outcomes by 15% Leveraged Optimizely to meet patient needs, achieving a customer satisfaction score of 93% measured through surveys Managed a team of 7 product specialists on Kanban Boards; which decreased product development cycle time by 21% Streamlined supply chain processes with C3 Al, reducing procurement costs by 16% Product Analyst Duolingo 2014 2016 Pittsburgh, PA Implemented Microsoft Azure analytics to boost the app's engagement, leading to a 47% upsurge in new user sign-ups Enhanced average time to resolve bugs to less than 3 minutes using Jira to automate tracking and reporting Developed a precise CTA across social media ads that resulted in a 37% boost in new app downloads Conducted AB testing on new Ul elements which resulted in a 41% jump in user engagement",
        "education": "Bachelor's degree Computer Science Carnegie Mellon University 2010 2014 Pittsburgh, PA",
        "personal information": "nbishop@emailcom (123) 456-7890 Hershey, PA Linkedln githubcom"
    }
}

    
    
    

parsed = parse_resume(data)
print(json.dumps(parsed, indent=4))

from langchain.chains.query_constructor.schema import AttributeInfo

standard_code_metadata = [
    AttributeInfo(name="source", description="Legal document source title", type="string"),
    AttributeInfo(name="title", description="Major division", type="string"),
    AttributeInfo(name="subtitle", description="Subdivision under title", type="string"),
    AttributeInfo(name="chapter", description="Chapter division under subdivision", type="string"),
    AttributeInfo(name="article", description="Article division under Chapter", type="string"),
    AttributeInfo(name="subchapter", description="Subchapter division under chapter", type="string"),
    AttributeInfo(name="part", description="Part division under subchapter", type="string"),
    AttributeInfo(name="section", description="Individual legal section", type="string"),
    AttributeInfo(name="page", description="Page number in source document", type="integer")
]

standard_code2_metadata = [
    AttributeInfo(name="source", description="Legal document source title", type="string"),
    AttributeInfo(name="title", description="Major division", type="string"),
    AttributeInfo(name="subtitle", description="Subdivision under title", type="string"),
    AttributeInfo(name="chapter", description="Chapter division under subdivision", type="string"),
    AttributeInfo(name="subchapter", description="Subchapter division under chapter", type="string"),
    AttributeInfo(name="part", description="Part division under subchapter", type="string"),
    AttributeInfo(name="article", description="Article division under Part", type="string"),
    AttributeInfo(name="section", description="Individual legal section", type="string"),
    AttributeInfo(name="page", description="Page number in source document", type="integer")
]

METADATA_FIELDS = {
        "Texas Occupations Code": standard_code2_metadata,
       
        "Texas Business and Commerce Code": standard_code_metadata,

        "Texas Business Organizations Code": standard_code_metadata,
        
        "Texas Agriculture Code": standard_code_metadata,
        
        "Texas Alcoholic Beverage Code": standard_code_metadata,
        
        "Texas Civil Practice And Remedies Code": standard_code_metadata,
        
        "Texas Education Code": standard_code_metadata,
        
        "Texas Election Code": standard_code_metadata,
        
        "Texas Estates Code": standard_code_metadata,

        "Texas Finance Code": standard_code_metadata,

        "Texas Human Resources Code": standard_code_metadata,

        "Texas Labor Code": standard_code_metadata,

        "Texas Natural Resources Code": standard_code_metadata,
        
        "Texas Parks And Wild Life Code": standard_code_metadata,

        "Texas Property Code": standard_code_metadata,

        "Texas Penal Code": standard_code_metadata,

        "Texas Water Code": standard_code_metadata,

        "Texas Code of Criminal Procedure": standard_code2_metadata,

        "Texas Utilities Code": standard_code_metadata,

        "Auxiliary Water Laws": standard_code2_metadata,

        "Texas Government Code": standard_code2_metadata,

        "Texas Local Government Code": standard_code_metadata,
        
        "Texas Tax Code": standard_code_metadata,

        "Texas Insurance Code": standard_code2_metadata,

        "Texas Family Code":standard_code2_metadata,

        "Rules of Judicial Education": [
                    AttributeInfo(name="source", description="The source document title, for example 'Rules of Judicial Education'.", type="string"),
                    AttributeInfo(name="rule", description="The main rule heading, such as 'RULE 14. TRAINING ON DUTIES REGARDING BAIL.'.", type="string"),
                    AttributeInfo(name="page", description="The page number in the source document.", type="integer")
                ],

        "Texas Rules of Evidence": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Texas Rules of Evidence'.",
                    type="string"
                ),
                AttributeInfo(
                    name="article",
                    description="The main article division, such as 'Article VI. Witnesses' or 'Article IX. Authentication and Identification'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The specific rule within an article, such as 'Rule 613. Witness's Prior Statement and Bias or Interest' or 'Rule 901. Authenticating or Identifying Evidence'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Texas Rules of Appellate Procedure": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Texas Rules of Appellate Procedure'.",
                    type="string"
                ),
                AttributeInfo(
                    name="section",
                    description="The main section division, such as 'Section One: General Provisions'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The main rule heading, such as 'Rule 1. Scope of Rule; Local Rules of Courts of Appeals'.",
                    type="string"
                ),
                AttributeInfo(
                    name="sub-rule",
                    description="A numbered subdivision under a rule, such as '1.1. Scope'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Texas Administrative Law Handbook": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Texas Administrative Law Handbook'.",
                    type="string"
                ),
                AttributeInfo(
                    name="chapter",
                    description="The main chapter division, such as 'Adjudication'.",
                    type="string"
                ),
                AttributeInfo(
                    name="section",
                    description="The section within a chapter, such as 'Initiating a Contested Case'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Federal Rules of Evidence": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Federal Rules of Evidence'.",
                    type="string"
                ),
                AttributeInfo(
                    name="article",
                    description="The main article division, such as 'ARTICLE IX. AUTHENTICATION AND IDENTIFICATION'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The specific rule within an article, such as 'Rule 901. Authenticating or Identifying Evidence'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Federal Rules of Criminal Procedure": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Federal Rules of Criminal Procedure'.",
                    type="string"
                ),
                AttributeInfo(
                    name="title",
                    description="The main title division, such as 'TITLE IV. ARRAIGNMENT AND PREPARATION FOR TRIAL'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The main rule heading within a title, such as 'Rule 17. Subpoena'.",
                    type="string"
                ),
                AttributeInfo(
                    name="part",
                    description="A subdivision or paragraph within a rule, such as '(c) Producing Documents and Objects.'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Federal Rules of Civil Procedure": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Federal Rules of Civil Procedure'.",
                    type="string"
                ),
                AttributeInfo(
                    name="title",
                    description="The main title division, such as 'TITLE VI. TRIALS'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The specific rule within a title, such as 'Rule 44.1. Determining Foreign Law'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Federal Rules of Bankruptcy Procedure": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Federal Rules of Bankruptcy Procedure'.",
                    type="string"
                ),
                AttributeInfo(
                    name="part",
                    description="The main part division, such as 'PART IX—GENERAL PROVISIONS'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The specific rule within a part, such as 'Rule 9017. Evidence'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Federal Rules of Appellate Procedure": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Federal Rules of Appellate Procedure'.",
                    type="string"
                ),
                AttributeInfo(
                    name="title",
                    description="The main title division, such as 'TITLE VII. GENERAL PROVISIONS'.",
                    type="string"
                ),
                AttributeInfo(
                    name="rule",
                    description="The specific rule within a title, such as 'Rule 41. Mandate: Contents; Issuance and Effective Date; Stay'.",
                    type="string"
                ),
                AttributeInfo(
                    name="topic",
                    description="A subdivision or paragraph within a rule, such as '(a) Contents.'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "Ethical Guidelines for Mediators": [
                AttributeInfo(
                    name="source",
                    description="The source document title, for example 'Ethical Guidelines for Mediators'.",
                    type="string"
                ),
                AttributeInfo(
                    name="title",
                    description="The main title or top-level division, such as 'GUIDELINES'.",
                    type="string"
                ),
                AttributeInfo(
                    name="section",
                    description="The section within the guidelines, such as 'Confidentiality'.",
                    type="string"
                ),
                AttributeInfo(
                    name="page",
                    description="The page number in the source document.",
                    type="integer"
                )
            ],
        "JCIT Technology Standards": [
                AttributeInfo(name="source", description="The source document title, for example 'JCIT Technology Standards'.", type="string"),
                AttributeInfo(name="chapter", description="The main chapter division, such as '1 Introduction'.", type="string"),
                AttributeInfo(name="topic", description="The topic within a chapter, such as '1.2 Versions'.", type="string"),
                AttributeInfo(name="subtopic", description="The more specific subtopic within a topic, such as '4.1.2 Filing Types'.", type="string"),
                AttributeInfo(name="page", description="The page number in the source document.", type="integer")
            ],
        "Statewide Rules Governing Electronic Filing in Criminal Cases": [
                AttributeInfo(name="source", description="The source document title, for example 'Statewide Rules Governing Electronic Filing in Criminal Cases'.", type="string"),
                AttributeInfo(name="part", description="The main part division, such as 'Part 1. General Provisions'.", type="string"),
                AttributeInfo(name="rule", description="The specific rule within a part, such as 'Rule 1.1 Scope'.", type="string"),
                AttributeInfo(name="page", description="The page number in the source document.", type="integer")
            ],

        "Texas Rules of Disciplinary Procedure": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="part", description="Chapter division (e.g., 'Part 1: General Rules')", type="string"),
            AttributeInfo(name="section", description="Section within part (e.g., '1.01. Citation:')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Texas Rules of Civil Procedure": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="part", description="Major division (e.g., 'PART I - GENERAL RULES')", type="string"),
            AttributeInfo(name="section", description="Section division under part (e.g., 'Section 1. General Rules' or 'SECTION I. PROCEDURES')", type="string"),
            AttributeInfo(name="subsection", description="Subsection under section (e.g., 'A. EVIDENCE', 'B. Discovery')", type="string"),
            AttributeInfo(name="rule", description="Individual rule (e.g., 'RULE 1. OBJECTIVE OF RULES', 'RULE 75a. FILING EXHIBITS')", type="string"),
            AttributeInfo(name="sub-rule", description="Sub-rule under main rule (e.g., '192.1 Forms of Discovery', '176.2 Required Actions')", type="string"),
            AttributeInfo(name="topic", description="Topic heading or subject within rule/sub-rule", type="string"),
            AttributeInfo(name="notes", description="Notes and Comments section providing context or change explanations", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Judicial Branch Certification Commission Rules": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., '2.0 Powers, Duties, and Responsibilities', '1.0 General Provisions')", type="string"),
            AttributeInfo(name="sub-rule", description="Sub-rule under main rule (e.g., '2.3 Powers and Duties of the Administrative Director', '1.1 Applicability of These Rules')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Education Rules on Guardianship, Alternatives to Guardianship, and Supports and Services for Proposed Wards and Wards": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., 'Rule 1. Authority', 'Rule 4. Approved Programs')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Judicial Bypass Rules under Ch. 33 of the Family Code": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., '2.0 Powers, Duties, and Responsibilities', '1.0 General Provisions')", type="string"),
            AttributeInfo(name="sub-rule", description="Sub-rule under main rule (e.g., '2.3 Powers and Duties of the Administrative Director', '1.1 Applicability of These Rules')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Procedural Rules for the State Commission on Judicial Conduct": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., 'Rule 1. Authority', 'Rule 4. Approved Programs')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Rules for Magistrates in Inmate Litigation and Litigation Involving Certain Civilly Committed Individuals": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., 'Rule 1. Authority', 'Rule 4. Approved Programs')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
        "Texas Rules of Judicial Administration": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="rule", description="Main rule division (e.g., '2.0 Powers, Duties, and Responsibilities', '1.0 General Provisions')", type="string"),
            AttributeInfo(name="sub-rule", description="Sub-rule under main rule (e.g., '2.3 Powers and Duties of the Administrative Director', '1.1 Applicability of These Rules')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
         "Texas Disciplinary Rules of Professional Conduct": [
            AttributeInfo(name="source", description="The Title of the Legal Book", type="string"),
            AttributeInfo(name="chapter", description="Chapter division (e.g., 'I. CLIENT-LAWYER RELATIONSHIP', 'Preamble')", type="string"),
            AttributeInfo(name="rule", description="Individual rule (e.g., 'Rule 1.15 Safekeeping Property', 'Rule 1.01 Competent and Diligent Representation', \"A Lawyer's Responsibilities\")", type="string"),
            AttributeInfo(name="comment", description="Comment section under rule providing explanation or context (e.g., 'Comment', 'Accepting Employment', 'Competent and Diligent Representation')", type="string"),
            AttributeInfo(name="page", description="Page number in source document", type="integer")
        ],
         "United States Constitution": [
                                AttributeInfo(
                                    name="source",
                                    description="The source document title, for example 'United States Constitution'.",
                                    type="string"
                                ),
                                AttributeInfo(
                                    name="article/amendment",
                                    description="The main constitutional division, such as 'Preamble', 'Article I', 'Article II', or 'Amendment II'.",
                                    type="string"
                                ),
                                AttributeInfo(
                                    name="section",
                                    description="The section within an article or amendment, such as 'Section 1.' or 'Section 3.'. Leave blank if the text is a standalone amendment without sections.",
                                    type="string"
                                ),
                                AttributeInfo(
                                    name="page",
                                    description="The page number in the source document.",
                                    type="integer"
                                ) ],
        "Texas Constitution": [
                                    AttributeInfo(
                                        name="source",
                                        description="The source document title, for example 'Texas Constitution'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="article",
                                        description="The main constitutional division ' Article V Judicial Department',' Article VI Suffrage' or 'Article XVI General Provisions'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="section",
                                        description="The section within an article, such as 'Sec. 1. Classes Of Persons Not Allowed To Vote.'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="page",
                                        description="The page number in the source document.",
                                        type="integer"
                                    )
                                ],
        "Uniform Format Manual for Texas Reporters Records": [
                                    AttributeInfo(
                                        name="source",
                                        description="The source document title, for example 'Uniform Format Manual for Texas Reporters Records'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="section",
                                        description="The main section of the manual, such as 'Section 1 - Uniform Terminology' or another top-level division.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="subsection",
                                        description="The subdivision within a section, such as a sub-section under the main section, such as '2.19 Placement of Page Heading.'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="topic",
                                        description="The specific topic or subject heading within a section or subsection, such as '(a) Court Reporter.'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="page",
                                        description="The page number in the source document.",
                                        type="integer"
                                    )
                                ],
    "Texas Code of Judicial Conduct": [
                                    AttributeInfo(
                                        name="source",
                                        description="The source document title, for example 'Texas Code of Judicial Conduct'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="canon",
                                        description="The main canon division, such as 'Canon 6: Compliance with the Code of Judicial Conduct'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="rule",
                                        description="A subdivision, paragraph, or lettered provision within a canon, such as 'C. Justices of the Peace and Municipal Court Judges.'.",
                                        type="string"
                                    ),
                                    AttributeInfo(
                                        name="page",
                                        description="The page number in the source document.",
                                        type="integer"
                                    )
                                ]
}
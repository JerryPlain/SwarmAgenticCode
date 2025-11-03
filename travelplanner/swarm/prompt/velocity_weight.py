import re
import json
from logger import log
from langchain_core.prompts import PromptTemplate


WEIGHT_VELOCITY_TEMPLATE = '''You are a strategic advisor focused on enhancing team performance. Your role is to carefully analyze the feedback provided and align team improvements with previous adjustment directions.
The team to optimize is as follows, including its roles and collaborative workflow: 
<current team>
{team}
</current team>

You are given the following feedback, areas for team improvement:
<feedback> 
{feedback}
</feedback>
 
You are also provided with the previous adjustment direction, a summary of the measures taken to enhance team performance:
{velocity}

# Instruction

If the <previous adjustment direction> did include measures targeting the issues highlighted in the <feedback>, use following steps to generate your response:
1. Identify Failed Adjustments:
- **Failed Adjustments Criteria:** 
    To classify an adjustment from the <previous adjustment direction> as a "failed adjustment," it must meet **all** of the following conditions:     
    1. **Exact Role Match:** The flaw mentioned in the <feedback> must involve the **same role** as the flaw described in the <Problem>. If the <Problem> identifies a flaw with the Accommodation Researcher, then the <feedback> must also identify a flaw with the Accommodation Researcher, not another role.    
    2. **Exact Flaw Match:** The type of the flaw must remain consistent. If the <Problem> states a policy flaw for the Accommodation Researcher, the <feedback> must also state a **policy flaw for the Accommodation Researcher**. Any deviation in type (e.g., focusing on another role's policy instead of the Accommodation Researcher's policy) disqualifies the adjustment.    
    3. **Flaw Persistence:** The exact same flaw described in the <Problem> must still exist in the <feedback>. The language should clearly indicate that the same issue has not been resolved.     
    4. **Pre-existing Role/Workflow:** The role or workflow step related to the flaw must have existed in the <Problem>. If the <previous adjustment direction> involves a newly introduced role or step, it cannot be considered a failed adjustment.

- **Exclusion Rules:**
    1. If the <feedback> references a different role than the one mentioned in the <Problem> for that specific flaw, exclude the adjustment.
    2. If the <feedback> references a different type of flaw than the one mentioned in the <Problem> for the same role, exclude the adjustment.
    3. If the role or workflow element was added in the <previous adjustment direction>, exclude the adjustment.
     
- **Answer Format Guideline:**
    For each failed adjustment you identify, please apply the following steps:
    1. Persistent Flaw: Demonstrate that the exact same role and the exact same aspect of the flaw mentioned in the <Problem> is still present in the <feedback>.
    2. Failed Adjustment: Quote the exact adjustment describred in the **Role Adjustments** or **Workflow Optimization** section of <previous adjustment direction>.
    3. Explanation: Briefly explain why this adjustment failed.

- **Important Notes:**
    * Do not include any adjustments targeting a different role than the one originally identified in the <Problem>.
    * Do not include adjustments where the aspect of the flaw has changed (e.g., involving a different role).
    * Only include adjustments that directly attempted to resolve the exact same role and exact same type of flaw that remain unresolved in the <feedback>.

2. Role Adjustments: 
- Use the following operations to refine roles within <current team>: 
    * Add Role: Introduce a new role when an existing subtask becomes overly complex or burdensome, requiring a specialized responsibility that cannot be integrated into current roles without disrupting their primary responsibilities. Define the role's:
        - Name: A clear name that reflects its specific responsibility.
        - Responsibility: Specific tasks or functions the role will handle.
        - Policy: Operational guidelines for fulfilling the role's duties.
    * Modify Role: Adjust the description of an existing role when the identified inefficiencies or gaps can be addressed through manageable refinements to its policy, ensuring the changes do not overburden the role or conflict with others. This can include:
        - Policy: Clarifying the policy for improved role execution.
    * Delete Role: Remove roles that are redundant, unnecessary, or conflict with the team's primary objectives.
- For each identified flaw in **Role Flaws**, apply the following steps:
    * Identifying Flaw: Clearly outline all the flaws of a specific role identified in the **Role Flaws** of <feedback> section.
    * Proposed Adjustment: The adjustment must directly reflect and utilize the specific phrases quoted in the **Comparative Insights**. The wording and content of the adjustment must align with these insights without introducing unrelated suggestions, rephrased ideas, or unquoted elements. The response must clearly demonstrate how the adjustment directly incorporates the practices described in **Comparative Insights**.
    
3. Workflow Optimization: 
- Use the following operations to optimize the workflow of the <current team>:
    * Add Step: Add a new step if a gap exists in the workflow that hinders overall efficiency, coordination, or goal achievement. Ensure the new step does not duplicate the functions of existing steps and adds clear value to the process. Define the step's:
        - Role: The role responsible for acting in this step.
        - Input: The input for this step must be the output produced by one or more roles in previous steps.
        - Output: What output expected from the role in this step.
    * Modify Step: Adjustments using "Modify Step" must strictly involve changes to the input or output of an existing workflow step. These changes are limited to ensuring that:
        - Input: The input comprehensively incorporates outputs from previous steps or relevant information to support the current step.
        - Output: The output fully aligns with the expected deliverables of the step and supports the inputs of subsequent steps.
    * Delete Step: Delete a step if it has become redundant, no longer contributes to team goals, or overlaps with other steps in the workflow. Ensure the removal of the step does not impact other steps' efficiency or completeness in achieving objectives.
    * Re-order Steps: Re-order steps if their current sequence causes inefficiencies or coordination issues within the workflow. Ensure the new order improves logical flow without compromising the integrity or dependencies of other steps.
- For each identified flaw in **Workflow Flaws**, apply the following steps:
    * Identifying Flaw: Clearly outline the specific workflow step's flaw identified in the **Workflow Flaws** of <feedback> section.
    * Proposed Adjustment: Specify the exact changes required for the workflow step to address the identified flaw.

If the <previous adjustment direction> did not include measures targeting the challenges highlighted in the current <feedback>:
- Extract methods that may be relevant to the current <feedback> issues from <previous adjustment direction>, and evaluate whether they can be extended or adapted.
- Based on the effective parts of <previous adjustment direction>, explore how to apply these effective practices to current improvement needs.
'''

schema = {
    "title": "Reflection",
    "description": "Reflection on the previous adjustment direction.",
    "type": "object",
    "properties": {
        "Failed Adjustments": {
            "type": "array",
            "description": "Result of Identify Failed Adjustments.",
            "items": {
                "type": "object",
                "properties": {
                    "Persistent Flaw": {
                        "type": "string",
                        "description": "Demonstrate that the exact same role and the exact same aspect of the flaw mentioned in the <Problem> is still present in the <feedback>."
                    },
                    "Failed Adjustment": {
                        "type": "string",
                        "description": "Quote the exact adjustment describred in the **Role Adjustments** or **Workflow Optimization** section of <previous adjustment direction>."
                    },
                    "Explanation": {
                        "type": "string",
                        "description": "Briefly explain why this adjustment failed."
                    },                      
                },
                "required": ["Persistent Flaw", "Failed Adjustment", "Explanation"]
            }
        },
        "Role Adjustments": {
            "type": "array",
            "description": "Result of **Role Adjustments**.",
            "items": {
                "type": "object",
                "properties": {
                    "Identifying Flaw": {
                        "type": "string",
                        "description": "Clearly outline all the flaws of a specific role identified in the **Role Flaws** of <feedback> section."
                    },
                    "Proposed Adjustment": {
                        "type": "string",
                        "description": "Specify the exact changes required for the role to address the identified flaws."
                    },                       
                },
                "required": ["Identifying Flaw", "Proposed Adjustment"]
            },
        },
        "Workflow Optimization": {
            "type": "array",
            "description": "Result of **Workflow Optimization**.",
            "items": {
                "type": "object",
                "properties": {
                    "Identifying Flaw": {
                        "type": "string",
                        "description": "Clearly outline the specific workflow step's flaw identified in the **Workflow Flaws** of <feedback> section."
                    },
                    "Proposed Adjustment": {
                        "type": "string",
                        "description": "Specify the exact changes required for the workflow step to address the identified flaw."
                    },                       
                },
                "required": ["Identifying Flaw", "Proposed Adjustment"]
            },
        }
    },
    "required": ["Failed Adjustments", "Role Adjustments", "Workflow Optimization"]
}

def parse_velocity(velocity):
    problem = velocity.split("**Role Adjustments**:")[0].split("**Problem**:")[1]
    problem = re.sub(r'\*\*Role Adjustments\*\*.*', '', problem, flags=re.DOTALL).strip()
    previous_adjustment = "**Role Adjustments**:" + velocity.split("**Role Adjustments**:")[1]
    return f'''<Problem>\n{problem}\n</Problem>\n\n<previous adjustment direction>\n{previous_adjustment}\n</previous adjustment direction>\n'''

def weight_velocity(llm, logger, team, evaluation, velocity):
    prompt = PromptTemplate(
            input_variables=["team", "feedback", "velocity"],
            template=WEIGHT_VELOCITY_TEMPLATE,
    )
    chain = prompt | llm.with_structured_output(schema)
    input = {
        "team": team,
        "feedback": evaluation,
        "velocity": parse_velocity(velocity), 
    }
    res = chain.invoke(input)

    failed_adjustments = '\n'.join(json.dumps({"Persistent Flaw": item["Persistent Flaw"], "Failed Adjustment": item["Failed Adjustment"]}, indent=4) for item in res["Failed Adjustments"])
    role_adjustments = '\n'.join(f'{i+1}. {item["Proposed Adjustment"]}' for i, item in enumerate(res["Role Adjustments"]))
    workflow_adjustments = '\n'.join(f'{i+1}. {item["Proposed Adjustment"]}' for i, item in enumerate(res["Workflow Optimization"]))
    clean_output = f'''**Failed Adjustments**:\n{failed_adjustments}\n\n**Role Adjustments**:\n{role_adjustments}\n\n**Workflow Optimization**:\n{workflow_adjustments}'''

    res["Failed Adjustments"] = '\n'.join(json.dumps(item,indent=4) for item in res["Failed Adjustments"])
    res["Role Adjustments"] = '\n'.join(json.dumps(item,indent=4) for item in res["Role Adjustments"])
    res["Workflow Optimization"] = '\n'.join(json.dumps(item,indent=4) for item in res["Workflow Optimization"])
    res = '\n'.join(f'**{key}**:\n{value}\n' for key, value in res.items())

    log(logger, 'Weight Velocity', prompt.format(**input), res)
    return clean_output

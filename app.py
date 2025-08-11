import streamlit as st
from openai import OpenAI
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import json
import time
import datetime
import uuid
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum

load_dotenv()
client = OpenAI()

class ActionType(Enum):
    CLICK = "click"
    TYPE = "type"
    EXTRACT = "extract"
    WAIT = "wait"
    NAVIGATE = "navigate"
    FINISHED = "finished"

class SelectorType(Enum):
    ID = "id"
    CLASS = "class"
    XPATH = "xpath"
    TEXT = "text"
    CSS = "css"

class BrowserAction:
    def __init__(self, 
                 action_type: ActionType,
                 selector_type: Optional[SelectorType] = None,
                 selector: Optional[str] = None,
                 input_value: Optional[str] = None,
                 requires_user_input: bool = False,
                 user_prompt: Optional[str] = None,
                 timeout: int = 10000):
        self.action_type = action_type
        self.selector_type = selector_type
        self.selector = selector
        self.input_value = input_value
        self.requires_user_input = requires_user_input
        self.user_prompt = user_prompt
        self.timeout = timeout

class BrowserAgent:
    def __init__(self):
        self.playwright = sync_playwright().start()
        self.browser = self.playwright.chromium.launch(headless=False)
        self.context = self.browser.new_context()
        self.page = self.context.new_page()
        self.current_url = ""

    def close(self):
        self.context.close()
        self.browser.close()
        self.playwright.stop()

    def navigate_to_url(self, url: str) -> bool:
        try:
            self.page.goto(url)
            self.current_url = url
            return True
        except Exception as e:
            st.error(f"Navigation error: {str(e)}")
            return False

    def _get_selector(self, selector_type: SelectorType, selector: str) -> str:
        if selector_type == SelectorType.ID:
            return f"#{selector}"
        elif selector_type == SelectorType.CLASS:
            return f".{selector}"
        elif selector_type == SelectorType.XPATH:
            return selector
        elif selector_type == SelectorType.TEXT:
            return f"text={selector}"
        else:
            return selector

    def find_and_click(self, selector_type: SelectorType, selector: str, timeout: int = 10000) -> bool:
        try:
            full_selector = self._get_selector(selector_type, selector)
            self.page.click(full_selector, timeout=timeout)
            return True
        except Exception as e:
            st.error(f"Click error: {str(e)}")
            return False

    def find_and_type(self, selector_type: SelectorType, selector: str, text: str, timeout: int = 10000) -> bool:
        try:
            full_selector = self._get_selector(selector_type, selector)
            self.page.fill(full_selector, text, timeout=timeout)
            return True
        except Exception as e:
            st.error(f"Type error: {str(e)}")
            return False

    def extract_content(self, selector_type: SelectorType, selector: str, timeout: int = 10000) -> Optional[str]:
        try:
            full_selector = self._get_selector(selector_type, selector)
            element = self.page.wait_for_selector(full_selector, timeout=timeout)
            return element.inner_text() if element else None
        except Exception as e:
            st.error(f"Extract error: {str(e)}")
            return None

    def wait_for_element(self, selector_type: SelectorType, selector: str, timeout: int = 10000) -> bool:
        try:
            full_selector = self._get_selector(selector_type, selector)
            self.page.wait_for_selector(full_selector, timeout=timeout)
            return True
        except Exception as e:
            st.error(f"Wait error: {str(e)}")
            return False

    def get_current_url(self) -> str:
        return self.current_url
        
    def get_page_content(self) -> str:
        try:
            html_content = self.page.content()
            soup = BeautifulSoup(html_content, 'html.parser')
            for script in soup(["script", "style", "meta", "svg"]):
                script.extract()
            return str(soup)
        except Exception as e:
            st.error(f"Error extracting page content: {str(e)}")
            return ""

class TaskPlanner:
    def __init__(self):
        self.session_id = str(uuid.uuid4())[:8]
        self.log_file = "llm_interactions.json"
        self.system_prompt = """
        You are a browser automation expert that helps execute web navigation tasks one step at a time.
        Your role is to determine the NEXT SINGLE action to take based on the current state and user's request.
        
        For each request:
        1. Consider what the user is trying to achieve
        2. Look at the current browser state (if provided)
           - The current_url shows the page the user is on
           - The page_content contains the HTML content of the webpage with script and style tags removed
           - Use this HTML content to identify elements by their attributes, structure, and visible text
        3. Determine the single most appropriate next action
        4. If you need user input, request it
        
        Available action_types:
        - "navigate": Go to a URL (input_value=URL)
        - "click": Click an element
        - "type": Enter text into a field
        - "extract": Get text from an element
        - "wait": Wait for an element to appear
        - "finished": Task is complete, no more actions needed
        
        Available selector_types:
        - "id": Use element ID
        - "class": Use CSS class
        - "xpath": Use XPath
        - "text": Find by exact text
        - "css": Use CSS selector
        
        Respond in JSON format with this exact structure for the NEXT SINGLE action to take:
        {
            "action_type": "navigate|click|type|extract|wait|finished",
            "selector_type": "id|class|xpath|text|css" (optional),
            "selector": "element_selector" (optional),
            "input_value": "text_to_type_or_url" (optional),
            "requires_user_input": boolean,
            "user_prompt": "Question to ask user" (if requires_user_input=true),
            "timeout": milliseconds (default=10000),
            "explanation": "Brief explanation of what this action accomplishes",
        }
        
        For the first action, if it makes sense to visit a website, use action_type="navigate".
        When navigating to websites, make sure that you're sure that the website URL actually exists.
        If the task is complete, use action_type="finished" and include a summary of what was accomplished.
        
        Examples:
        
        1. First action for "Check weather in New York":
        {
            "action_type": "navigate",
            "input_value": "https://weather.com",
            "requires_user_input": false,
            "explanation": "Navigating to Weather.com to check the forecast",
        }
        
        2. First action for "Order pizza from Domino's":
        {
            "action_type": "navigate",
            "input_value": "https://dominos.com",
            "requires_user_input": false,
            "explanation": "Navigating to Domino's Pizza website", 
        }
        
        3. Task completion example:
        {
            "action_type": "finished",
            "explanation": "Successfully added basketball to cart on Amazon",
        }
        """

        with open(self.log_file, "w") as f:
            f.write("")


    def log_llm_interaction(self, interaction_type: str, messages: List[Dict], response_content: str, result: Dict) -> None:
        try:
            timestamp = datetime.datetime.now().isoformat()
            log_entry = {
                "timestamp": timestamp,
                "session_id": self.session_id,
                "interaction_type": interaction_type,
                "messages": messages,
                "response": response_content,
                "parsed_result": result
            }
            
            try:
                with open(self.log_file, "r") as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            except FileNotFoundError:
                logs = []
            
            logs.append(log_entry)
            
            with open(self.log_file, "w") as f:
                json.dump(logs, f, indent=2)
                
        except Exception as e:
            st.error(f"Error logging LLM interaction: {str(e)}")
    
    def get_next_action(self, user_input: str, current_state: Optional[Dict] = None, conversation_history: Optional[List[Dict]] = None) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
        else:
            messages.append({"role": "user", "content": user_input})
            
        if current_state:
            messages.append({"role": "user", "content": f"Current state: {json.dumps(current_state)}"})

        with open("dump.json", "w") as f:
            json.dump(current_state, f)
        
        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        result = json.loads(response_content)

        self.log_llm_interaction("get_next_action", messages, response_content, result)

        with open("dump.json", "w+") as f:
            f.write("\n\n\n" + json.dumps(result, indent=4))
        
        if conversation_history is not None:
            conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        return result

    def handle_error(self, error: str, current_state: Dict, conversation_history: List[Dict]) -> Dict:
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]
        
        if conversation_history:
            messages.extend(conversation_history)
            
        messages.append({"role": "user", "content": f"Error occurred: {error}\nCurrent state: {json.dumps(current_state)}\nPlease provide an alternative approach."})        

        response = client.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            response_format={"type": "json_object"}
        )

        response_content = response.choices[0].message.content
        result = json.loads(response_content)
        
        self.log_llm_interaction("handle_error", messages, response_content, result)
        
        conversation_history.append({"role": "user", "content": f"Error occurred: {error}"})
        conversation_history.append({"role": "assistant", "content": json.dumps(result)})
        
        return result

def execute_browser_action(agent: BrowserAgent, action_data: Dict) -> Tuple[bool, Optional[str]]:
    action_type = ActionType(action_data["action_type"]) if "action_type" in action_data else None
    
    if action_type == ActionType.FINISHED:
        return True, "finished"
    
    action = BrowserAction(
        action_type=action_type,
        selector_type=SelectorType(action_data.get("selector_type")) if action_data.get("selector_type") else None,
        selector=action_data.get("selector"),
        input_value=action_data.get("input_value"),
        requires_user_input=action_data.get("requires_user_input", False),
        user_prompt=action_data.get("user_prompt"),
        timeout=action_data.get("timeout", 10000)
    )
    
    if action.action_type == ActionType.NAVIGATE:
        success = agent.navigate_to_url(action.input_value)
        return success, None
    elif action.action_type == ActionType.CLICK:
        success = agent.find_and_click(action.selector_type, action.selector, action.timeout)
        return success, None
    elif action.action_type == ActionType.TYPE:
        success = agent.find_and_type(action.selector_type, action.selector, action.input_value, action.timeout)
        return success, None
    elif action.action_type == ActionType.EXTRACT:
        content = agent.extract_content(action.selector_type, action.selector, action.timeout)
        return content is not None, content
    elif action.action_type == ActionType.WAIT:
        success = agent.wait_for_element(action.selector_type, action.selector, action.timeout)
        return success, None
    
    return False, None

def main():
    st.title("Browser Agent")
    st.write("What do you want to do?")

    # Initialize session state
    if 'agent' not in st.session_state:
        st.session_state.agent = None
    if 'task_planner' not in st.session_state:
        st.session_state.task_planner = TaskPlanner()
    if 'user_task' not in st.session_state:
        st.session_state.user_task = None
    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []
    if 'current_action' not in st.session_state:
        st.session_state.current_action = None
    if 'user_inputs' not in st.session_state:
        st.session_state.user_inputs = {}
    if 'finished' not in st.session_state:
        st.session_state.finished = False
    if 'task_progress' not in st.session_state:
        st.session_state.task_progress = ""
    if 'extracted_content' not in st.session_state:
        st.session_state.extracted_content = None

    if not st.session_state.user_task:
        user_input = st.text_input("What would you like me to do?", key="task_input")
        if user_input:
            st.session_state.user_task = user_input
            st.session_state.conversation_history = [{"role": "user", "content": user_input}]
            st.session_state.agent = BrowserAgent()
            st.rerun()

    if st.session_state.user_task and not st.session_state.finished:
        if st.session_state.task_progress:
            st.info(st.session_state.task_progress)
            
        if st.session_state.extracted_content:
            st.write("Extracted content:")
            st.code(st.session_state.extracted_content)
        
        if not st.session_state.current_action:
            with st.spinner("Thinking about next step..."):
                current_state = {
                    "current_url": st.session_state.agent.get_current_url() if st.session_state.agent else "",
                    "page_content": st.session_state.agent.get_page_content() if st.session_state.agent else ""
                }
                st.session_state.current_action = st.session_state.task_planner.get_next_action(
                    st.session_state.user_task,
                    current_state,
                    st.session_state.conversation_history
                )
        
        if "explanation" in st.session_state.current_action:
            st.write(st.session_state.current_action["explanation"])
            
        if "task_progress" in st.session_state.current_action:
            st.session_state.task_progress = st.session_state.current_action["task_progress"]

        requires_input = st.session_state.current_action.get("requires_user_input", False)
        user_prompt = st.session_state.current_action.get("user_prompt", "")
        
        if requires_input and user_prompt not in st.session_state.user_inputs:
            user_response = st.text_input(user_prompt, key="user_response")
            if st.button("Submit"):
                st.session_state.user_inputs[user_prompt] = user_response
                st.session_state.current_action["input_value"] = user_response
                st.session_state.conversation_history.append({"role": "user", "content": f"User provided input: {user_response}"})
                st.rerun()
        else:
            with st.spinner("Executing action..."):
                success, extracted = execute_browser_action(st.session_state.agent, st.session_state.current_action)
                
                if success:
                    if st.session_state.current_action.get("action_type") == "finished":
                        st.success("Task completed!")
                        st.session_state.finished = True
                        if st.button("Start New Task"):
                            if st.session_state.agent:
                                st.session_state.agent.close()
                            st.session_state.agent = None
                            st.session_state.user_task = None
                            st.session_state.conversation_history = []
                            st.session_state.current_action = None
                            st.session_state.user_inputs = {}
                            st.session_state.finished = False
                            st.session_state.task_progress = ""
                            st.session_state.extracted_content = None
                            st.rerun()
                    else:
                        if extracted:
                            st.session_state.extracted_content = extracted
                            st.session_state.conversation_history.append({"role": "user", "content": f"Extracted content: {extracted}"})
                            
                        st.session_state.current_action = None
                        st.rerun()
                else:
                    st.error("Error occurred during action execution")
                    with st.spinner("Finding alternative approach..."):
                        current_state = {
                            "current_url": st.session_state.agent.get_current_url(),
                            "page_content": st.session_state.agent.get_page_content()
                        }
                        st.session_state.current_action = st.session_state.task_planner.handle_error(
                            "Action failed",
                            current_state,
                            st.session_state.conversation_history
                        )
                        st.rerun()

if __name__ == "__main__":
    main()

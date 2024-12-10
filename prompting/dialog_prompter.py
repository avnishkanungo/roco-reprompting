import os 
import json
import pickle 
from openai import OpenAI
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
from os.path import join
from typing import List, Tuple, Dict, Union, Optional, Any

from rocobench.subtask_plan import LLMPathPlan
from rocobench.rrt_multi_arm import MultiArmRRT
from rocobench.envs import MujocoSimEnv, EnvState, visualize_voxel_scene 
from .feedback import FeedbackManager
from .parser import LLMResponseParser
from copy import deepcopy

# assert os.path.exists("openai_key.json"), "Please put your OpenAI API key in a string in robot-collab/openai_key.json"
load_dotenv()

PATH_PLAN_INSTRUCTION="""
[Path Plan Instruction]
Each <coord> is a tuple (x,y,z) for gripper location, follow these steps to plan:
1) Decide target location (e.g. an object you want to pick), and your current gripper location.
2) Plan a list of <coord> that move smoothly from current gripper to the target location.
3) The <coord>s must be evenly spaced between start and target.
4) Each <coord> must not collide with other robots, and must stay away from table and objects.  
[How to Incoporate [Enviornment Feedback] to improve plan]
    If IK fails, propose more feasible step for the gripper to reach. 
    If detected collision, move robot so the gripper and the inhand object stay away from the collided objects. 
    If collision is detected at a Goal Step, choose a different action.
    To make a path more evenly spaced, make distance between pair-wise steps similar.
        e.g. given path [(0.1, 0.2, 0.3), (0.2, 0.2. 0.3), (0.3, 0.4. 0.7)], the distance between steps (0.1, 0.2, 0.3)-(0.2, 0.2. 0.3) is too low, and between (0.2, 0.2. 0.3)-(0.3, 0.4. 0.7) is too high. You can change the path to [(0.1, 0.2, 0.3), (0.15, 0.3. 0.5), (0.3, 0.4. 0.7)] 
    If a plan failed to execute, re-plan to choose more feasible steps in each PATH, or choose different actions.
[How to Incoporate [Rough Overarching Plan from Human Verifier] ,[Human Verifier Observations and Feedback] and [Human Input on Failure] to further improve and optimize planning process]
    These inputs are included to integrate Human in the Loop approach to optimize the planning process by providing more context aout the task at hand
    Alice and Bob need to take into account the [Rough Overarching Plan from Human Verifier] and [Human Verifier Observations and Feedback] when discussing the what move each of the robotic agent needs to do.
    Alice and Bob need to take into account [Human Input on Failure] when there is a failure in implemneting or parsing the plan.
    All three inputs are important context and need to be taken into consideration whenever they are non-empty in the prompt.
    When [Human Verifier Observations and Feedback] mentions that an observation about the curent setup made by any agent is incorrect, the feedback provided by the human verifier takes precedence and the LLM needs to provide a plan based on that.
    If [Human Verifier Observations and Feedback] contains a command with "EXECUTE", ignore the response and pass last item in the array as the response.
"""

HUMAN_VERIFIER_INSTRUCTIONS = """
[How to Incoporate [Rough Overarching Plan from Human Verifier] ,[Human Verifier Observations and Feedback] and [Human Input on Failure] to further improve and optimize planning process]
    These inputs are included to integrate Human in the Loop approach to optimize the planning process by providing more context aout the task at hand.
    When there is a conflict or difference between the plans suggested by the two agents for a round, use the round instructions for that round from the [Rough Overarching Plan from Human Verifier] to proceed further.
    Alice and Bob need to take into account the [Rough Overarching Plan from Human Verifier] and [Human Verifier Observations and Feedback] when discussing the what move each of the robotic agent needs to do.
    Alice and Bob need to take into account [Human Input on Failure] when there is a failure in implementing or parsing the plan.
    All three inputs are important context and need to be taken into consideration whenever they are non-empty in the prompt.
    The latest value in the [Human Verifier Observations and Feedback] is the most relevant to the current round.
    When [Human Verifier Observations and Feedback] mentions that an observation about the curent setup made by any agent is incorrect, the feedback provided by the human verifier takes precedence and the LLM needs to provide a plan based on that.
    If [Human Verifier Observations and Feedback] contains a command with "EXECUTE", ignore the response and pass last item in the array as the response.
"""

class DialogPrompter:
    """
    Each round contains multiple prompts, query LLM once per each agent 
    """
    def __init__(
        self,
        env: MujocoSimEnv,
        parser: LLMResponseParser,
        feedback_manager: FeedbackManager, 
        max_tokens: int = 512,
        debug_mode: bool = False,
        use_waypoints: bool = False,
        robot_name_map: Dict[str, str] = {"panda": "Bob"},
        num_replans: int = 3, 
        max_calls_per_round: int = 10,
        use_history: bool = True,  
        use_feedback: bool = True,
        temperature: float = 0,
        llm_source: str = "gpt-4"
        
    ):
        self.max_tokens = max_tokens
        self.debug_mode = debug_mode
        self.use_waypoints = use_waypoints
        self.use_history = use_history
        self.use_feedback = use_feedback
        self.robot_name_map = robot_name_map
        self.robot_agent_names = list(robot_name_map.values())
        self.num_replans = num_replans
        self.env = env
        self.feedback_manager = feedback_manager
        self.parser = parser
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.human_feedback = []
        self.obs_based_human_input = []
        self.max_calls_per_round = max_calls_per_round 
        self.temperature = temperature
        self.llm_source = llm_source
        assert llm_source in ["gpt-4", "gpt-3.5-turbo", "claude","meta/llama-3.1-405b-instruct"], f"llm_source must be one of [gpt4, gpt-3.5-turbo, claude, meta/llama-3.1-405b-instruct], got {llm_source}"

    def compose_system_prompt(
        self, 
        obs: EnvState, 
        agent_name: str,
        chat_history: List = [], # chat from previous replan rounds
        current_chat: List = [],  # chat from current round, this comes AFTER env feedback 
        feedback_history: List = [],
        rough_plan: List = []
    ) -> str:
        action_desp = self.env.get_action_prompt()

        if len(rough_plan)>0:
            action_desp += HUMAN_VERIFIER_INSTRUCTIONS

        if self.use_waypoints:
            action_desp += PATH_PLAN_INSTRUCTION
        agent_prompt = self.env.get_agent_prompt(obs, agent_name)
        
        round_history = self.get_round_history() if self.use_history else ""

        execute_feedback = ""
        if len(self.failed_plans) > 0: #potentially where we should look to add human feedback, print out the failed plans for human review and pass the review comments in the execute feedback variable
            execute_feedback = "Plans below failed to execute, improve them to avoid collision and smoothly reach the targets:\n"
            execute_feedback += "\n".join(self.failed_plans) + "\n"

        chat_history = "[Previous Chat]\n" + "\n".join(chat_history) if len(chat_history) > 0 else ""
        system_prompt = f"{action_desp}\n{round_history}\n{execute_feedback}\n{agent_prompt}\n{chat_history}" 
        
        if len(rough_plan)>0:    
            system_prompt = f"{action_desp}\n{round_history}\n[Rough Overarching Plan from Human Verifier]\n{rough_plan}\n{execute_feedback}\n{agent_prompt}\n{chat_history}\n[Human Verifier Observations and Feedback]\n{self.obs_based_human_input}" 
        
        if self.use_feedback and len(feedback_history) > 0:
            system_prompt += "\n".join(feedback_history)
        
        if len(current_chat) > 0:
            system_prompt += "[Current Chat]\n" + "\n".join(current_chat) + "\n"
        # combines action_desp, round_history, execute_feedback, agent_prompt, chat_history, feedback_history and current_chat to generate the system prompt
        # keeps changing on the basis of the values that are received each time promp_one_dialog_round is called.
        return system_prompt 

    def get_round_history(self):
        if len(self.round_history) == 0:
            return ""
        ret = "[History]\n"
        for i, history in enumerate(self.round_history):
            ret += f"== Round#{i} ==\n{history}\n"
        ret += f"== Current Round ==\n"
        return ret
    
    def prompt_one_round(self, obs: EnvState, save_path: str = "", step: int=0, rough_plan:list=[]): 
        
        if len(rough_plan)>0:
            if step > 0:
                env_disp = deepcopy(self.env)
                env_disp.physics.data.qpos[:] = self.env.physics.data.qpos[:].copy()
                env_disp.physics.forward()
                env_disp.render_point_cloud = True
                obs = env_disp.get_obs()
                visualize_voxel_scene(
                    obs.scene,
                    save_img=os.path.join(save_path, f"step_{step}_initial_state.jpg")
                    )
                
                while True:
                    initial_human_feedback = input("Enter any comments that the human verifier wants to pass to the LLM on the current plan (press Enter to append, or 'q' to quit without comments): ")
                    if initial_human_feedback == "q":
                        self.obs_based_human_input.append("Follow next round instruction from overarching plan.")
                        print("Default Feedback appended. Exiting.")
                        break
                    elif initial_human_feedback:
                        self.obs_based_human_input.append(initial_human_feedback)
                        print("Human Feedback appended. Exiting.")
                        break  # Exit the loop after successfully receiving feedback
        
        plan_feedbacks = []
        chat_history = [] 
        for i in range(self.num_replans):
            final_agent, final_response, agent_responses = self.prompt_one_dialog_round(
                obs,
                chat_history,
                plan_feedbacks,
                replan_idx=i,
                save_path=save_path,
                rough_plan=rough_plan
            )
            # based on feedback this will allow 5 replans if there is any issue that comes up during validation,
            # this validation will be successful if the plan follows the path rules correctly, it does not take into account
            # whether the plan actually does the task, it just validates the semantics. Not the quatily of the path or its ability
            # to achieve said task.
            chat_history += agent_responses
            parse_succ, parsed_str, llm_plans = self.parser.parse(obs, final_response) 

            curr_feedback = "None"
            if not parse_succ:  
                curr_feedback = f"""
This previous response from [{final_agent}] failed to parse!: '{final_response}'
{parsed_str} Re-format to strictly follow [Action Output Instruction]!"""
                ready_to_execute = False  
            
            else:
                ready_to_execute = True
                for j, llm_plan in enumerate(llm_plans): 
                    ready_to_execute, env_feedback = self.feedback_manager.give_feedback(llm_plan)        
                    if not ready_to_execute:
                        curr_feedback = env_feedback
                        break
            plan_feedbacks.append(curr_feedback)
            tosave = [
                {
                    "sender": "Feedback",
                    "message": curr_feedback,
                },
                {
                    "sender": "Action",
                    "message": (final_response if not parse_succ else llm_plans[0].get_action_desp()),
                },
            ]
            timestamp = datetime.now().strftime("%m%d-%H%M")
            fname = f'{save_path}/replan{i}_feedback_{timestamp}.json'
            json.dump(tosave, open(fname, 'w')) 

            if ready_to_execute: 
                break  
            else:
                print(curr_feedback)
        self.latest_chat_history = chat_history
        return ready_to_execute, llm_plans, plan_feedbacks, chat_history
   
    def prompt_one_dialog_round(
        self, 
        obs, 
        chat_history, 
        feedback_history, 
        replan_idx=0,
        save_path='data/',
        rough_plan: List = []
        ):
        """
        keep prompting until an EXECUTE is outputted or max_calls_per_round is reached.
        prompts will alternate between the robots.
        """
        
        agent_responses = []
        usages = []
        dialog_done = False 
        num_responses = {agent_name: 0 for agent_name in self.robot_agent_names}
        n_calls = 0

        while n_calls < self.max_calls_per_round:
            for agent_name in self.robot_agent_names:
                system_prompt = self.compose_system_prompt(
                    obs, 
                    agent_name,
                    chat_history=chat_history,
                    current_chat=agent_responses,
                    feedback_history=feedback_history,
                    rough_plan=rough_plan   
                    ) 
                
                agent_prompt = f"You are {agent_name}, your response is:"
                if n_calls == self.max_calls_per_round - 1:
                    agent_prompt = f"""
You are {agent_name}, this is the last call, you must end your response by incoporating all previous discussions and output the best plan via EXECUTE. 
Your response is:
                    """
                response, usage = self.query_once( ### gets the prompts and calls the function which calls the LLM API
                    system_prompt, 
                    user_prompt=agent_prompt, 
                    max_query=3,
                    )
    
                tosave = [ 
                    {
                        "sender": "SystemPrompt",
                        "message": system_prompt,
                    },
                    {
                        "sender": "UserPrompt",
                        "message": agent_prompt,
                    },
                    {
                        "sender": agent_name,
                        "message": response
                    },
                    usage,
                ]
                timestamp = datetime.now().strftime("%m%d-%H%M")
                fname = f'{save_path}/replan{replan_idx}_call{n_calls}_agent{agent_name}_{timestamp}.json'
                json.dump(tosave, open(fname, 'w'))  

                num_responses[agent_name] += 1
                # strip all the repeated \n and blank spaces in response: 
                pruned_response = response.strip()
                # pruned_response = pruned_response.replace("\n", " ")
                agent_responses.append(
                    f"[{agent_name}]:\n{pruned_response}"
                    )
                usages.append(usage)
                n_calls += 1
                if 'EXECUTE' in response:
                    if replan_idx > 0 or all([v > 0 for v in num_responses.values()]):
                        dialog_done = True
                        break
 
                if self.debug_mode:
                    dialog_done = True
                    break
            
            if dialog_done:
                break
 
        # response = "\n".join(response.split("EXECUTE")[1:])
        # print(response)  
        return agent_name, response, agent_responses

    def query_once(self, system_prompt, user_prompt, max_query):
        response = None
        usage = None   
        # print('======= system prompt ======= \n ', system_prompt)
        # print('======= user prompt ======= \n ', user_prompt)

        if self.debug_mode: 
            response = "EXECUTE\n"
            for aname in self.robot_agent_names:
                action = input(f"Enter action for {aname}:\n")
                response += f"NAME {aname} ACTION {action}\n"
            return response, dict()

        client = OpenAI(
                        base_url = "https://integrate.api.nvidia.com/v1",
                        api_key = os.environ.get("NVIDIA_API_KEY") 
                    ) #### Change this after the LLAMA test

        for n in range(max_query):
            print('querying {}th time'.format(n))
            try:
                ## response = openai.ChatCompletion.create, revert to this after the test
                response = client.chat.completions.create(
                    model=self.llm_source, 
                    messages=[
                        # {"role": "user", "content": ""},
                        {"role": "system", "content": system_prompt+user_prompt},                                    
                    ],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    ).dict()
                usage = response['usage']
                response = response['choices'][0]['message']["content"]
                # response_final = response
                print('======= response ======= \n ', response)
                print('======= usage ======= \n ', usage)
                break
            except Exception as e:
                print("API error: {e}, try again")
            continue
        # breakpoint()
        return response, usage
    
    def post_execute_update(self, obs_desp: str, execute_success: bool, parsed_plan: str): ## add a human feedback parameter
        ## create a global variable array here which will store in the human feedback, the human feedback post one step
        ## will be passed as a parameter for the post execute update.
        if execute_success: 
            # clear failed plans, count the previous execute as full past round in history
            self.failed_plans = []
            chats = "\n".join(self.latest_chat_history)
            self.round_history.append(
                f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}"
                #f"[Chat History]\n{chats}\n[Executed Action]\n{parsed_plan}\n[Verifier Input]\n{self.human_feedback}"
            )
        else:
            # while True:
            #     human_verifier_input = input("Enter any comments that the human verifier wants to pass to the LLM on plan failure (press Enter to append, or 'q' to quit without comments): ")
            #     if human_verifier_input == "q":
            #         break
            #     elif human_verifier_input:
            #         self.human_feedback.append(human_verifier_input)
            #         print("Feedback appended. Exiting.")
            #         break  # Exit the loop after successfully receiving feedback            
            self.failed_plans.append(
                f"[Failed Executed Action]{parsed_plan}"
                # f"[Failed Executed Action]{parsed_plan}\n[Human Input on Failure]\n{self.human_feedback}"
            )
        return 

    def post_episode_update(self):
        # clear for next episode
        self.round_history = []
        self.failed_plans = [] 
        self.latest_chat_history = []
        self.obs_based_human_input = []
## Prototype
Check the notebook ```final_proto.ipynb``` for the prototype of our work

## Running with UI (Chainlit)

There is a User Interface version of the prototype. Check ```app_team_user_proxy.py`` for the UI version. The original UI example link is here: [Link](https://github.com/microsoft/autogen/tree/main/python/samples/agentchat_chainlit). 

### AutoGen and Chainlit Environment
```shell
pip install -U chainlit autogen-agentchat autogen-ext[openai] pyyaml
```

### Run
```shell
chainlit run app_team_user_proxy.py -h
```
** Terminate with Crtl+C in the bash.
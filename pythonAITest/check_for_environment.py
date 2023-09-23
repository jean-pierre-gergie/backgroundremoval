




conda_env_list = [
    "MachineLearning          C:\\Users\\jpg\\anaconda3\\envs\\MachineLearning",
    "OpenCV                   C:\\Users\\jpg\\anaconda3\\envs\\OpenCV",
    "OpenCV2                  C:\\Users\\jpg\\anaconda3\\envs\\OpenCV2",
    "adnan                    C:\\Users\\jpg\\anaconda3\\envs\\adnan",
    "ai                       C:\\Users\\jpg\\anaconda3\\envs\\ai",
    "batrounguide             C:\\Users\\jpg\\anaconda3\\envs\\batrounguide",
    "botframework             C:\\Users\\jpg\\anaconda3\\envs\\botframework",
    "django_proj_t            C:\\Users\\jpg\\anaconda3\\envs\\django_proj_t",
    "eurisko                  C:\\Users\\jpg\\anaconda3\\envs\\eurisko",
    "flaskappTest             C:\\Users\\jpg\\anaconda3\\envs\\flaskappTest",
    "forlinux                 C:\\Users\\jpg\\anaconda3\\envs\\forlinux",
    "jwyn                     C:\\Users\\jpg\\anaconda3\\envs\\jwyn",
    "minimalreq               C:\\Users\\jpg\\anaconda3\\envs\\minimalreq",
    "minimalreq2              C:\\Users\\jpg\\anaconda3\\envs\\minimalreq2",
    "myenv                    C:\\Users\\jpg\\anaconda3\\envs\\myenv",
    "peer1                    C:\\Users\\jpg\\anaconda3\\envs\\peer1",
    "ratelimit                C:\\Users\\jpg\\anaconda3\\envs\\ratelimit",
    "roboapp                  C:\\Users\\jpg\\anaconda3\\envs\\roboapp"
]

# Extract only the environment names
environment_names = [env.split()[0] for env in conda_env_list]

# Print the list of environment names
print(environment_names)
#
import os
import subprocess



for env in environment_names:
    try:
        # Activate the environment
        activate_cmd = f"conda activate {env}"
        subprocess.call(activate_cmd, shell=True)

        # Check for TensorFlow
        result = subprocess.run(["python", "-c", "import tensorflow as tf; print(tf.__version__)"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, text=True)
        if "ModuleNotFoundError" not in result.stderr:
            print(f"TensorFlow is installed in the '{env}' environment.")
    except Exception as e:
        print(f"Error checking environment '{env}': {e}")
    finally:
        # Deactivate the environment
        deactivate_cmd = "conda deactivate"
        subprocess.call(deactivate_cmd, shell=True)












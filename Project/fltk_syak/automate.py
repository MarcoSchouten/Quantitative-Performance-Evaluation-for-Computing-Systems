# pip install google-cloud-container

import csv
import itertools
import json
import os
import subprocess
import time
import uuid

from kubernetes import client, config, watch

# Global Variables
PROJECT_ID = 'cs4215-325512'
ZONE = 'us-central1-c'
CLUSTER_ID = 'fltk-cluster'
NODES_PER_POOL = 4
DEFAULT_NAMESPACE = "default"
CUSTOM_NAMESPACE = "test"
KUBERNETES_DASHBOARD = "app.kubernetes.io/name=kubernetes-dashboard,app.kubernetes.io/instance=kubernetes-dashboard"
TENSORBOARD_DASHBOARD = "fltk.service=fl-extractor"
TENSORBOARD_POD_NAME = ""
KUBERNETES_POD_NAME = ""
ORCHESTRATOR_LABEL = "fltk.service=fl-server"
JOB_LABEL = "app=fltk-worker,controller-name=pytorch-operator"
FMT = '%Y-%m-%dT%H:%M:%SZ'
JSON_FILE = "./configs/tasks/example_arrival_config.json"
NODE_POOLS = ['default-pool', 'fltk-pool']

# configurations

# 1: 2-Factorial design, 2: Full factorial design, 3: Test
DESIGN_CHOICE = 1
NUM_REPLICATIONS = 2

if DESIGN_CHOICE == 1:
    # 2-Factorial design
    EXECUTOR_CORES_OPTIONS = [1, 3]
    DATA_PARALLEL_OPTIONS = [1, 4]
    BATCH_SIZE_OPTIONS = [32, 256]
    LEARNING_RATE_OPTIONS = [0.01, 0.05]
    MAX_EPOCH_OPTIONS = [5, 20]

elif DESIGN_CHOICE == 2:
    # Full factorial design
    EXECUTOR_CORES_OPTIONS = [1, 2, 3]
    DATA_PARALLEL_OPTIONS = [1, 2, 4]
    BATCH_SIZE_OPTIONS = [32, 64, 128, 256]
    LEARNING_RATE_OPTIONS = [0.01, 0.02, 0.05]
    MAX_EPOCH_OPTIONS = [5, 10, 20]

else:
    # Test setup
    EXECUTOR_CORES_OPTIONS = [1]
    DATA_PARALLEL_OPTIONS = [1]
    BATCH_SIZE_OPTIONS = [32]
    LEARNING_RATE_OPTIONS = [0.01]
    MAX_EPOCH_OPTIONS = [5]


def execute(*popenargs, **kwargs):
    if 'stdout' in kwargs:
        raise ValueError('stdout argument not allowed, it will be overridden.')
    process = subprocess.Popen(stdout=subprocess.PIPE, shell=True, *popenargs, **kwargs)
    for line in process.stdout:
        yield line.decode()
    process.stdout.close()
    retcode = process.wait()
    if retcode:
        cmd = kwargs.get("args")
        if cmd is None:
            cmd = popenargs[0]
    #     raise subprocess.CalledProcessError(retcode, cmd)


def start_gcp():
    # Authorize application: Uncomment the following code if running for first time
    # for path in execute(["gcloud auth application-default login"]):
    #     print(path, end="")

    cmd = "gcloud container clusters resize {} --node-pool {} --num-nodes {} --zone {} --quiet"

    for idx, node in enumerate(NODE_POOLS):
        for path in execute([cmd.format(CLUSTER_ID, node, NODES_PER_POOL, ZONE)]):
            print(path, end="")

    print(f'Running on node {NODE_POOLS}')


def first_setup():
    global KUBERNETES_POD_NAME, TENSORBOARD_POD_NAME

    # Get Kubernetes Dashboard Command
    ret = v1.list_namespaced_pod(namespace=DEFAULT_NAMESPACE,
                                 label_selector=KUBERNETES_DASHBOARD,
                                 watch=False)
    for p in ret.items:
        KUBERNETES_POD_NAME = p.metadata.name
        print(f'Execute:  kubectl -n {DEFAULT_NAMESPACE} port-forward {KUBERNETES_POD_NAME} 8443:8443')

    # Install Tensorboard
    for path in execute(["helm install extractor ./charts/extractor -f ./charts/fltk-values.yaml -n test"]):
        print(path, end="")

    # Check status of tensorboard
    for event in w.stream(func=v1.list_namespaced_pod,
                          namespace=CUSTOM_NAMESPACE,
                          label_selector=TENSORBOARD_DASHBOARD,
                          timeout_seconds=600):
        if event["object"].status.phase == "Running":
            w.stop()

    ret = v1.list_namespaced_pod(namespace=CUSTOM_NAMESPACE,
                                 label_selector=TENSORBOARD_DASHBOARD,
                                 watch=False)
    for p in ret.items:
        TENSORBOARD_POD_NAME = p.metadata.name
        print(f'Execute:  kubectl -n {CUSTOM_NAMESPACE} port-forward {TENSORBOARD_POD_NAME} 6006:6006')

    # Uninstall orchestrator
    for path in execute(["helm uninstall orchestrator -n test"]):
        print(path, end="")


def run_experiments():
    # Scale up GCP
    start_gcp()

    # Initialize setup
    first_setup()

    num_experiments = 0

    # Experiments
    header = ['exp_no',
              'cores',
              'parallelization',
              'batch_size',
              'learning_rate',
              'max_epoch',
              'start_time',
              'end_time',
              'job_name']

    if not os.path.isdir("./results"):
        os.mkdir("./results")

    if not os.path.isfile("./results/log.csv"):
        with open("./results/log.csv", 'w') as f:
            writer = csv.writer(f)
            writer.writerow(header)

    with open("./results/log.csv", 'a') as f:
        writer = csv.writer(f)
        # writer.writerow(header)

        for combination in list(itertools.product(
                *[EXECUTOR_CORES_OPTIONS, DATA_PARALLEL_OPTIONS, BATCH_SIZE_OPTIONS, LEARNING_RATE_OPTIONS,
                  MAX_EPOCH_OPTIONS])):

            print(f'Running experiment no {num_experiments + 1}')

            # Update combination in json file
            update_json(combination)

            # Docker build
            for path in execute(["DOCKER_BUILDKIT=1 docker build . --tag gcr.io/cs4215-325512/fltk"]):
                print(path, end="")
            print('Built Docker image')

            # Docker push
            for path in execute(["docker push gcr.io/cs4215-325512/fltk"]):
                print(path, end="")
            print('Pushed Docker image')

            for replication in range(NUM_REPLICATIONS):
                # Install orchestrator
                for path in execute(["helm install orchestrator ./charts/orchestrator -f ./charts/fltk-values.yaml -n "
                                     "test"]):
                    print(path, end="")

                # Monitor orchestrator
                for event in w.stream(func=v1.list_namespaced_pod,
                                      namespace=CUSTOM_NAMESPACE,
                                      label_selector=ORCHESTRATOR_LABEL,
                                      timeout_seconds=600):
                    if event["object"].status.phase == "Running":
                        w.stop()
                        print('Orchestrator Started')

                time.sleep(10)

                # Monitor jobs
                for event in w.stream(func=v1.list_namespaced_pod,
                                      namespace=CUSTOM_NAMESPACE,
                                      label_selector=JOB_LABEL,
                                      timeout_seconds=2400):
                    if event["object"].status.phase == "Error" or event["object"].status.phase == "Succeeded":
                        w.stop()
                        print("Jobs Completed")

                # Get start and end time
                process = subprocess.check_output(
                    'kubectl get pytorchjobs -n test -o jsonpath="{.items[0].metadata.name}"',
                    shell=True)
                jobname = process.decode()

                process = subprocess.check_output(
                    'kubectl get pytorchjobs -n test -o jsonpath="{.items[0].status.completionTime}"',
                    shell=True)
                # completion_time = datetime.strptime(process.decode(), FMT)
                completion_time = process.decode()

                process = subprocess.check_output('kubectl get pytorchjobs -n test -o jsonpath="{.items['
                                                  '0].status.startTime}"',
                                                  shell=True)
                # start_time = datetime.strptime(process.decode(), FMT)
                start_time = process.decode()

                # dur = (completion_time - start_time).seconds

                # print(f'Job duration: {dur}')

                (EXECUTOR_CORES, DATA_PARALLEL, BATCH_SIZE, LEARNING_RATE, MAX_EPOCH) = combination
                num_experiments += 1
                row = [num_experiments, EXECUTOR_CORES, DATA_PARALLEL,
                       BATCH_SIZE, LEARNING_RATE, MAX_EPOCH, start_time, completion_time, jobname]
                writer.writerow(row)

                # Delete all pytorch jobs
                for path in execute(["kubectl delete pytorchjobs --all -n test"]):
                    print(path, end="")

                # Uninstall orchestrator
                for path in execute(["helm uninstall orchestrator -n test"]):
                    print(path, end="")

    # Clean up
    cleanup()

    # Scale down GCP
    stop_gcp()


def cleanup():
    # Take backup of extractor
    cmd = "kubectl cp --namespace test {}:/opt/federation-lab/logging ./logging"
    for path in execute([cmd.format(TENSORBOARD_POD_NAME)]):
        print(path, end="")

    # Remove the extractor
    for path in execute(["helm uninstall extractor -n test"]):
        print(path, end="")


def stop_gcp():
    cmd = "gcloud container clusters resize {} --node-pool {} --num-nodes 0 --zone {} --quiet"

    for idx, node in enumerate(NODE_POOLS):
        for path in execute([cmd.format(CLUSTER_ID, node, ZONE)]):
            print(path, end="")


def update_json(combination):
    (EXECUTOR_CORES, DATA_PARALLEL, BATCH_SIZE, LEARNING_RATE, MAX_EPOCH) = combination
    with open(JSON_FILE, 'r') as jf:
        task = json.load(jf)
        task[0]['jobClassParameters'][0]['systemParameters']['dataParallelism'] = str(DATA_PARALLEL)
        task[0]['jobClassParameters'][0]['systemParameters']['executorCores'] = str(EXECUTOR_CORES)
        task[0]['jobClassParameters'][0]['hyperParameters']['batchSize'] = str(BATCH_SIZE)
        task[0]['jobClassParameters'][0]['hyperParameters']['maxEpoch'] = str(MAX_EPOCH)
        task[0]['jobClassParameters'][0]['hyperParameters']['learningRate'] = str(LEARNING_RATE)

    tempfile = os.path.join(os.path.dirname(JSON_FILE), str(uuid.uuid4()))
    with open(tempfile, 'w') as tf:
        json.dump(task, tf, indent=4)

    os.replace(tempfile, JSON_FILE)


if __name__ == "__main__":
    # Configs can be set in Configuration class directly or using helper utility
    config.load_kube_config()
    v1 = client.CoreV1Api()
    w = watch.Watch()

    try:
        run_experiments()
    except:
        cleanup()
        stop_gcp()
    finally:
        print('Completed running')

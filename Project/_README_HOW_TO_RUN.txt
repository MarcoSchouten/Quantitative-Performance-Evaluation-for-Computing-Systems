

HOW TO RUN AN EXPERIMENT:

-1) WSL TERMINAL: cd  /mnt/c/Users/marco/test_setup/fltk-testbed 
0) change XML to run a specific config
1) build docker file
docker build . --tag gcr.io/qpecs-325510/fltk
2) push docker
docker push gcr.io/qpecs-325510/fltk
3) uninstall orchestrator
helm uninstall orchestrator -n test
4) kill all pytorch jobs
kubectl delete pytorchjobs --all -n test
5 install orchestrator
cd charts
helm install orchestrator ./orchestrator --namespace test -f fltk-values.yaml

5.5) make sure everything works 
kubectl get pods -n test

 6) kubectl get pytorchjobs -n test

7) leggi ID 

 8) kubectl get pytorchjobs -n test -o yaml trainjob-cb4ea058-4b8e-473c-adc2-867e481f1b1b

9) leggi dati

FOR DASHBOARD_______________
explorer.exe .
\\wsl$\Ubuntu-20.04\home\marcowsl\.kube
you will find the config file


 helm status kubernetes-dashboard
--> you get two other commands:

FIRST: export POD_NAME=$(kubectl get pods -n default -l "app.kubernetes.io/name=kubernetes-dashboard,app.kubernetes.io/instance=kubernetes-dashboard" -o jsonpath="{.items[0].metadata.name}")

SECOND:  kubectl -n default port-forward $POD_NAME 8443:8443

now this will work:
https://localhost:8443/



FOR TENSORBOARD _____________________________
1) helm uninstall extractor -n test
2) helm install extractor ./extractor -f fltk-values.yaml -n test
3) kubectl port-forward fl-extractor 6006:6006 -n test



TO SHUT DOWN ____________
1) helm uninstall orchestrator -n test
2) kubectl delete pytorchjobs --all -n test
3) remove nodes on RESIZE on google cloud console
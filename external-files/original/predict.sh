model_name=traffic-forecasting-model
file_name=input.json

CLUSTER_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
INGRESS_PORT=$(kubectl get svc istio-ingressgateway -n istio-system -o jsonpath='{.spec.ports[?(@.port==80)].nodePort}')

curl -v -H "Host: $model_name.kserve-test.svc.cluster.local" \
     -d @./$file_name \
     http://$CLUSTER_IP:$INGRESS_PORT/v1/models/$model_name:predict

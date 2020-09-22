try:
	from openvino.inference_engine import IECore
	from openvino.inference_engine import IENetwork
except:
	print('OpenVINO not installed')


def sync_inference(exec_net, image):	# runs syncronously (waits until one inference is done to start the next one)
	input_blob = next(iter(exec_net.inputs))
	result = exec_net.infer({input_blob: image})
	return result
 
def async_inference(exec_net, image, request_id=0):
	input_blob = next(iter(exec_net.inputs))
	exec_net.start_async(request_id, inputs={input_blob: image})
	return exec_net

def get_async_output(exec_net, request_id=0):
	output_blob = next(iter(exec_net.outputs))
	status = exec_net.requests[request_id].wait(-1)
	if status == 0:
		result = exec_net.requests[request_id].outputs[output_blob]
		return result

def load_to_IE(model_xml):

	# Getting the *.bin file location
	model_bin = model_xml[:-3]+"bin"
	
	#Loading the Inference Engine API
	ie = IECore()
	
	#Loading IR files
	net = IENetwork(model=model_xml, weights = model_bin)


	# Listing all the layers and supported layers
	cpu_extension_needed = False
	network_layers = net.layers.keys()
	supported_layer_map = ie.query_network(network=net,device_name="CPU")
	supported_layers = supported_layer_map.keys()

	# Checking if CPU extension is needed   
	for layer in network_layers:
		if layer in supported_layers:
			pass
		else:
			cpu_extension_needed =True
			print("CPU extension needed")
			break

	# Adding CPU extension
	if cpu_extension_needed:
		ie.add_extension(extension_path=cpu_ext, device_name="CPU")
		print("CPU extension added")
	else:
		print("CPU extension not needed")


	#Getting the supported layers of the network  
	supported_layer_map = ie.query_network(network=net, device_name="CPU")
	supported_layers = supported_layer_map.keys()
	
	# Checking for any unsupported layers, if yes, exit
	unsupported_layer_exists = False
	network_layers = net.layers.keys()
	for layer in network_layers:
		if layer in supported_layers:
			pass
		else:
			print(layer +' : Still Unsupported')
			unsupported_layer_exists = True
	if unsupported_layer_exists:
		print("Exiting the program.")
		exit(1)

	# Loading the network to the inference engine
	exec_net = ie.load_network(network=net, device_name="CPU")
	print("IR successfully loaded into Inference Engine.")
	return exec_net   #exec_net short for executable network

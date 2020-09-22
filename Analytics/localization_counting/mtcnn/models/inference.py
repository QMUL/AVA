try:
	#Loading the Inference Engine API
	from openvino.inference_engine import IECore
	from openvino.inference_engine import IENetwork
	from openvino.inference_engine import IEPlugin
	ie = IECore()
except:
	print('OpenVINO not installed')

import copy
import time 

def sync_inference(net, image, mode=None):	# runs syncronously (waits until one inference is done to start the next one)
	
	input_blob = 'images'
	if mode=='pnet':
		key = image.shape
		exec_net = net[str(key)]
		
	if mode in ['rnet','onet']:
		net.batch_size = image.shape[0]
		exec_net = ie.load_network(network=net, device_name="CPU")

	return exec_net.infer({input_blob: image})


def load_to_IE(model_xml, mode=None):

	# Getting the *.bin file location
	print(model_xml)
	model_bin = model_xml[:-3]+"bin"
	

	#Loading IR files
	net = IENetwork(model=model_xml, weights=model_bin)

	# Enable dynamic batchsize
	#plugin = IEPlugin(device='CPU')
	#plugin.set_config({'DYN_BATCH_ENABLED': 'YES'})

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

	if mode=='pnet':
		input_shapes = [
			[1, 3, 649, 1153],
			[1, 3, 460, 817],
			[1, 3, 326, 580],
			[1, 3, 231, 411],
			[1, 3, 164, 292],
			[1, 3, 117, 207],
			[1, 3, 83, 147],
			[1, 3, 59, 104],
			[1, 3, 42, 74],
			[1, 3, 30, 53],
			[1, 3, 21, 37],
			[1, 3, 15, 27],
		]
		exec_nets = {}
		input_blob = next(iter(net.inputs))
		for shape in input_shapes:
			net.reshape({input_blob: tuple(shape)})
			exec_net = ie.load_network(network=net, device_name="CPU")
			exec_nets[str(tuple(shape))] = exec_net
		return exec_nets
	else:
		return net


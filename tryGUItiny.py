from operator import itemgetter
from simple_file_checksum import get_checksum
import time
import PySimpleGUI as sg
import tinygrad
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear
import numpy as np
import pickle

def get_player():
	global saved_game, save
	for line in saved_game:
		if(line[0:6] == 'date="'):
			content = saved_game.readlines(10000) 
			saved_game = open(save)
			tag = content[0][8:-2]
			break
	return tag

def get_player_data():
	global saved_game, save
	tag = get_player()
	temp_prestige = 0
	temp_domestic = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	temp_techs = [0,0,0,0,0,0,0,0,0,0,0,0]
	allowed_factories = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1]
	for line in saved_game:
		if(line[0:4] == tag + "="):
			for line2 in saved_game:
				if(line2[0:12] == "	technology="):
					for line3 in saved_game:
						if(line3 == "	}\n"):
							break
						elif(line3.split('=')[0] == "		combustion_engine"):
							temp_techs[0] = 1
							allowed_factories[3] = 1
						elif(line3.split('=')[0] == "		electrical_power_generation"):
							temp_techs[1] = 1
						elif(line3.split('=')[0] == "		mechanical_production"):
							temp_techs[2] = 1
							allowed_factories[20] = 1
						elif(line3.split('=')[0] == "		interchangeable_parts"):
							temp_techs[3] = 1
						elif(line3.split('=')[0] == "		assembly_line"):
							temp_techs[4] = 1
						elif(line3.split('=')[0] == "		cheap_iron"):
							temp_techs[5] = 1
						elif(line3.split('=')[0] == "		cheap_steel"):
							temp_techs[6] = 1
						elif(line3.split('=')[0] == "		advanced_metallurgy"):
							temp_techs[7] = 1
						elif(line3.split('=')[0] == "		electric_furnace"):
							temp_techs[8] = 1
						elif(line3.split('=')[0] == "		inorganic_chemistry"):
							temp_techs[9] = 1
						elif(line3.split('=')[0] == "		electricity"):
							temp_techs[10] = 1
							allowed_factories[9] = 1
							allowed_factories[28] = 1
						elif(line3.split('=')[0] == "		synthetic_polymers"):
							temp_techs[11] = 1
							allowed_factories[0] = 1
						elif(line3.split('=')[0] == "		mechanized_mining"):
							allowed_factories[1] = 1
							allowed_factories[2] = 1
							allowed_factories[10] = 1
							allowed_factories[24] = 1 
							allowed_factories[26] = 1
						elif(line3.split('=')[0] == "		infiltration"):
							allowed_factories[4] = 1
						elif(line3.split('=')[0] == "		guild_based_production"):
							allowed_factories[5] = 1
						elif(line3.split('=')[0] == "		private_banks"):
							allowed_factories[6] = 1
						elif(line3.split('=')[0] == "		clipper_design"):
							allowed_factories[7] = 1
						elif(line3.split('=')[0] == "		water_wheel_power"):
							allowed_factories[11] = 1
						elif(line3.split('=')[0] == "		basic_chemistry"):
							allowed_factories[12] = 1
						elif(line3.split('=')[0] == "		organic_chemistry"):
							allowed_factories[8] = 1
							allowed_factories[13] = 1
						elif(line3.split('=')[0] == "		early_classical_theory_and_critique"):
							allowed_factories[14] = 1
							allowed_factories[17] = 1
							allowed_factories[19] = 1
							allowed_factories[21] = 1
						elif(line3.split('=')[0] == "		freedom_of_trade"):
							allowed_factories[18] = 1
						elif(line3.split('=')[0] == "		behaviorism"):
							allowed_factories[22] = 1
						elif(line3.split('=')[0] == "		publishing_industry"):
							allowed_factories[23] = 1
						elif(line3.split('=')[0] == "		steamers"):
							allowed_factories[25] = 1
						elif(line3.split('=')[0] == "		government_interventionism"):
							allowed_factories[27] = 1
				elif(line2[0:10] == '	prestige='):
					temp_prestige = line2.split('=')[1][:-1]
				elif(line2[0:22] == '	domestic_supply_pool='):
					for line3 in saved_game:
						if(line3 == "	}\n"):
							break
						elif(line3.split('=')[0] == "		ammunition"):
							temp_domestic[0] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		small_arms"):
							temp_domestic[1] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		artillery"):
							temp_domestic[2] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		canned_food"):
							temp_domestic[3] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		barrels"):
							temp_domestic[4] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		aeroplane"):
							temp_domestic[5] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		cotton"):
							temp_domestic[6] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		dye"):
							temp_domestic[7] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		wool"):
							temp_domestic[8] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		silk"):
							temp_domestic[9] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		coal"):
							temp_domestic[10] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		sulphur"):
							temp_domestic[11] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		iron"):
							temp_domestic[12] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		timber"):
							temp_domestic[13] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		tropical_wood"):
							temp_domestic[14] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		rubber"):
							temp_domestic[15] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		oils"):
							temp_domestic[16] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		precious_metal"):
							temp_domestic[17] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		steel"):
							temp_domestic[18] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		cement"):
							temp_domestic[19] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		machine_parts"):
							temp_domestic[20] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		glass"):
							temp_domestic[21] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		fuel"):
							temp_domestic[22] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		fertilizer"):
							temp_domestic[23] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		explosives"):
							temp_domestic[24] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		clipper_convoy"):
							temp_domestic[25] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		steamer_convoy"):
							temp_domestic[26] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		electric_gear"):
							temp_domestic[27] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		fabric"):
							temp_domestic[28] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		lumber"):
							temp_domestic[29] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		paper"):
							temp_domestic[30] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		cattle"):
							temp_domestic[31] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		fish"):
							temp_domestic[32] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		fruit"):
							temp_domestic[33] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		grain"):
							temp_domestic[34] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		tobacco"):
							temp_domestic[35] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		tea"):
							temp_domestic[36] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		coffee"):
							temp_domestic[37] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		opium"):
							temp_domestic[38] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		automobiles"):
							temp_domestic[39] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		telephones"):
							temp_domestic[40] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		wine"):
							temp_domestic[41] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		liquor"):
							temp_domestic[42] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		regular_clothes"):
							temp_domestic[43] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		luxury_clothes"):
							temp_domestic[44] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		furniture"):
							temp_domestic[45] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		luxury_furniture"):
							temp_domestic[46] = line3.split('=')[1][:-1]
						elif(line3.split('=')[0] == "		radio"):
							temp_domestic[47] = line3.split('=')[1][:-1]
					break
			break
	return temp_prestige, temp_techs, temp_domestic, allowed_factories

def combine_player_data():
	year = get_year()
	prestige, techs, domestic, allowed = get_player_data()
	world_demand = get_demand_to_supply() 
	prices = get_prices()

	newline = []
	#factoryInserter = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
	newline.append(float(year / 100))	#year
	newline.append(float(prestige))	#prestige
	for y in range(0, 12):
		newline.append(float(techs[y]))	#important techs
	for y in range(0, 48):
		newline.append(float(round(float(domestic[y])*10)/10))	#domestic
	for y in range(0, 48):
		newline.append(float(world_demand[y]))	#demand/supply
	for y in range(0, 48):
		newline.append(float(round(float(prices[y])*10)/10))	#prices
	for y in range(0, 30):
		newline.append(0)	#factory types
	return newline, allowed

def get_year():
	global saved_game, save
	with open(save) as f:
		year = int(f.readline()[6:][:4]) - 1835 
	return year 

def get_prices():
    global saved_game, save
    priceArray = []
    for line in saved_game:
        if(line[0:12] == "	price_pool="):
            content = saved_game.readlines(10000)
            saved_game = open(save)
            for x in range(1,49):
            	price = content[x][:-1]
            	hmm = price.split('=')
            	priceArray.append(hmm[1])
            break
    return priceArray

def get_demand_to_supply():
    #print(str(time.perf_counter()) + " get_demand_to_supply")
    global saved_game, save
    actual_sold = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    actual_sold_world = []
    real_demand = []
    divided = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    saved_game = open(save)
    for line in saved_game:
        if(line[0:13] == "	actual_sold="):    #include bottom, but not top
            content = saved_game.readlines(10000)   #cant copy objects in python...
            saved_game = open(save)
            for x in range(1,49):
            	hmm = content[x].split('=')
            	if(content[x] == "	}\n"):
            		break
            	elif(content[x].split('=')[0] == "		ammunition"):
            		actual_sold[0] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		small_arms"):
            		actual_sold[1] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		artillery"):
            		actual_sold[2] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		canned_food"):
            		actual_sold[3] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		barrels"):
            		actual_sold[4] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		aeroplane"):
            		actual_sold[5] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		cotton"):
            		actual_sold[6] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		dye"):
            		actual_sold[7] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		wool"):
            		actual_sold[8] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		silk"):
            		actual_sold[9] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		coal"):
            		actual_sold[10] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		sulphur"):
            		actual_sold[11] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		iron"):
            		actual_sold[12] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		timber"):
            		actual_sold[13] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		tropical_wood"):
            		actual_sold[14] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		rubber"):
            		actual_sold[15] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		oils"):
            		actual_sold[16] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		precious_metal"):
            		actual_sold[17] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		steel"):
            		actual_sold[18] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		cement"):
            		actual_sold[19] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		machine_parts"):
            		actual_sold[20] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		glass"):
            		actual_sold[21] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		fuel"):
            		actual_sold[22] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		fertilizer"):
            		actual_sold[23] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		explosives"):
            		actual_sold[24] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		clipper_convoy"):
            		actual_sold[25] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		steamer_convoy"):
            		actual_sold[26] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		electric_gear"):
            		actual_sold[27] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		fabric"):
            		actual_sold[28] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		lumber"):
            		actual_sold[29] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		paper"):
            		actual_sold[30] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		cattle"):
            		actual_sold[31] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		fish"):
            		actual_sold[32] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		fruit"):
            		actual_sold[33] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		grain"):
            		actual_sold[34] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		tobacco"):
            		actual_sold[35] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		tea"):
            		actual_sold[36] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		coffee"):
            		actual_sold[37] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		opium"):
            		actual_sold[38] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		automobiles"):
            		actual_sold[39] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		telephones"):
            		actual_sold[40] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		wine"):
            		actual_sold[41] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		liquor"):
            		actual_sold[42] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		regular_clothes"):
            		actual_sold[43] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		luxury_clothes"):
            		actual_sold[44] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		furniture"):
            		actual_sold[45] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		luxury_furniture"):
            		actual_sold[46] = hmm[1][:-1]
            	elif(content[x].split('=')[0] == "		radio"):
            		actual_sold[47] = hmm[1][:-1]
            break
    for line in saved_game:
        if(line[0:19] == "	actual_sold_world="):
            content = saved_game.readlines(10000)
            saved_game = open(save)
            for x in range(1,49):
            	price = content[x][:-1]
            	hmm = price.split('=')
            	actual_sold_world.append(hmm[1])
            break
    for line in saved_game:
        if(line[0:12] == "	real_demand"):
            content = saved_game.readlines(10000)
            saved_game = open(save)
            for x in range(1,49):
            	price = content[x][:-1]
            	hmm = price.split('=')
            	real_demand.append(hmm[1])
            break
    for x in range(0,48):
    	subcount = (float(real_demand[x]) / (float(actual_sold[x]) + float(actual_sold_world[x]))) * 100
    	divided[x] = round(subcount) /100
    return divided

factory_dict = {
	"0": "aeroplane_factory",
	"1": "ammunition_factory",
	"2": "artillery_factory",
	"3": "automobile_factory",
	"4": "barrel_factory",
	"5": "canned_food_factory",
	"6": "cement_factory",
	"7": "clipper_shipyard",
	"8": "dye_factory",
	"9": "electric_gear_factory",
	"10": "explosives_factory",
	"11": "fabric_factory",
	"12": "fertilizer_factory",
	"13": "fuel_refinery",
	"14": "furniture_factory",
	"15": "glass_factory",
	"16": "liquor_distillery",
	"17": "lumber_mill",
	"18": "luxury_clothes_factory",
	"19": "luxury_furniture_factory",
	"20": "machine_parts_factory",
	"21": "paper_mill",
	"22": "radio_factory",
	"23": "regular_clothes_factory",
	"24": "small_arms_factory",
	"25": "steamer_shipyard",
	"26": "steel_factory",
	"27": "synthetic_oil_factory",
	"28": "telephone_factory",
	"29": "winery"
}

def get_child_mine(parent, key): #(tinygradVersionOfNet, )
  obj = parent
  print(obj[1])
  for k in key.split('.'):
    print(k)
    if k.isnumeric():
      obj = obj[int(k)]
    elif isinstance(obj, dict):
      obj = obj[k]
    else:
      obj = getattr(obj, k)
  return obj

def my_unpickle(fb0):
  key_prelookup = {}
  class HackTensor:
    def __new__(cls, *args):
      #print(args)
      ident, storage_type, obj_key, location, obj_size = args[0][0:5] #ident is always storage, class(np.float32), filename(0-1124), device(cpu), parameters
      #assert ident == 'storage'

      #assert prod(args[2]) == obj_size
      ret = np.zeros(args[2], dtype=storage_type)
      key_prelookup[obj_key] = (storage_type, obj_size, ret, args[2], args[3])
      return ret

  class HackParameter:
    def __new__(cls, *args):
      #print(args)
      pass

  class Dummy:
    pass

  class MyPickle(pickle.Unpickler):
    def find_class(self, module, name):
      #print(module, name)
      if name == 'FloatStorage':
        return np.float32
      if name == 'LongStorage':
        return np.int64
      if name == 'HalfStorage':
        return np.float16
      if module == "torch._utils":
        if name == "_rebuild_tensor_v2":
          return HackTensor
        elif name == "_rebuild_parameter":
          return HackParameter
      else:
        try:
          return pickle.Unpickler.find_class(self, module, name)
        except Exception:
          return Dummy

    def persistent_load(self, pid):
      return pid
  return MyPickle(fb0).load(), key_prelookup

def fake_torch_load_zipped_mine(fb0, load_weights=True):
  import zipfile                                        #this is an import
  with zipfile.ZipFile(fb0, 'r') as myzip:              #this uses that import to extract a file
    with myzip.open('vickyFactories/data.pkl') as myfile:      
      ret = my_unpickle(myfile)   #this reconstructs the entire thing
      #print(ret[1])      #this is empty, but correct structure       
    if load_weights:
      for k,v in ret[1].items():  #for every data file(k), and every Tensor v(class, params, ([values], dtype), (thisLayerConfig), (followingLayerConfig))
        with myzip.open(f'vickyFactories/data/{k}') as myfile:
          #print(v[2])
          if v[2].dtype == "object":  #the dtype is the last element of v[2]
            print(f"issue assigning object on {k}")
            continue
          np.copyto(v[2], np.frombuffer(myfile.read(), v[2].dtype).reshape(v[3])) #the v[2] is an actual part of the ret(not a copy), anyway this reads the entire file(k), and reshapes the values based on the layer(v[3])
          #print(v[2])
  #print(ret[1])      #data is filled in
  return ret[0] #epoch, steps, pytorchVer, stateDict

class TinyNet:
    def __init__(self):
        self.net = [
            Linear(188, 3008, bias=True),
            lambda x: x.relu(),
            Linear(3008, 1504, bias=True),
            lambda x: x.relu(),
            Linear(1504, 1504, bias=True),
            lambda x: x.relu(),
            Linear(1504, 1504, bias=True),
            lambda x: x.relu(),
            Linear(1504, 1504, bias=True),
            lambda x: x.relu(),
            Linear(1504, 752, bias=True),
            lambda x: x.relu(),
            Linear(752, 1, bias=True)
        ]

    def __call__(self, x): 
    	return x.sequential(self.net)


#save="F:/VickySaves/save games/autosave.v2"
save_hash = 0

model = TinyNet()
dat = fake_torch_load_zipped_mine(open('vickyFactories.pt', "rb"))

tempTensor = Tensor(dat['fc1.weight'])
model.net[0].weight = tempTensor
tempTensor = Tensor(dat['fc1.bias'])
model.net[0].bias = tempTensor
tempTensor = Tensor(dat['fc2.weight'])
model.net[2].weight = tempTensor
tempTensor = Tensor(dat['fc2.bias'])
model.net[2].bias = tempTensor
tempTensor = Tensor(dat['fc3.weight'])
model.net[4].weight = tempTensor
tempTensor = Tensor(dat['fc3.bias'])
model.net[4].bias = tempTensor
tempTensor = Tensor(dat['fc3.weight'])
model.net[6].weight = tempTensor
tempTensor = Tensor(dat['fc3.bias'])
model.net[6].bias = tempTensor
tempTensor = Tensor(dat['fc3.weight'])
model.net[8].weight = tempTensor
tempTensor = Tensor(dat['fc3.bias'])
model.net[8].bias = tempTensor
tempTensor = Tensor(dat['fc4.weight'])
model.net[10].weight = tempTensor
tempTensor = Tensor(dat['fc4.bias'])
model.net[10].bias = tempTensor
tempTensor = Tensor(dat['fc5.weight'])
model.net[12].weight = tempTensor
tempTensor = Tensor(dat['fc5.bias'])
model.net[12].bias = tempTensor

layout = [[sg.In(size=(45,1), enable_events=True ,key='FOLDER', disabled=True), sg.FolderBrowse(key='BROWSE')],[sg.Multiline(size = (40,30), key="SHOW", font="Consolas")]]
window = sg.Window('Factory profitability', layout)

while True:
	event, values = window.read(timeout=1000)
	if event == sg.WIN_CLOSED: 
		break
	save = window.Element('FOLDER').get() + "/autosave.v2"
	if save != "/autosave.v2":
		window.Element('BROWSE').update(disabled=True)
	while save != "/autosave.v2":
		event, values = window.read(timeout=1000)
		if event == sg.WIN_CLOSED: 
			break
		old_hash = save_hash
		save_hash = get_checksum(save, algorithm="MD5")
		if(old_hash != save_hash):
			try:
				saved_game = open(save)
				array, factories = combine_player_data()
				saved_game = ""
				array[158] = 1
				line = Tensor(array)
				temp_array = []
				if(factories[0] == 1):
					temp_array.append([factory_dict["0"], str(round(model(line).numpy()[0]*1000)/1000)])

				for x in range(159,188):
					array[x-1] = 0
					array[x] = 1
					line = Tensor(array)
					if(factories[x-158] == 1):
						temp_array.append([factory_dict[str(x-158)], str(round(model(line).numpy()[0]*1000)/1000)])

				sorted_array = sorted(temp_array, key=itemgetter(1), reverse=True)
				megastring = ""
				for x in range(0, len(sorted_array)):
					megastring += sorted_array[x][0]
					for y in range(len(sorted_array[x][0])-28, 0):
						megastring += " "
					megastring += sorted_array[x][1]
					megastring += "\n"
				window.Element('SHOW').update(megastring)
			except Exception as e:
				window.Element('SHOW').update("no autosave in directory")


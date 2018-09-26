import argparse
import os
import re
import itertools

arg_parser = argparse.ArgumentParser(description='i/o options')
#arg_parser.add_argument("-i", "--input", type=str, help="Input file path", default=os.getcwd())
#arg_parser.add_argument("-o", "--output", type=str, help="Output file path", default=os.getcwd())
arg_parser.add_argument("-temp", "--template", type=str, help="Template file name", default="temp.inp")
arg_parser.add_argument("-para", "--para_list", type=str, help="Steps for equilibating charges", default="STEP:STEP.txt;Lvac:Lvac.txt;PARA3:para3_list.txt")
arg_parser.add_argument("-volt", "--volt_list", type=str, help="external voltage list file", default="")
arg_parser.add_argument("-ns", "--sim_time", type=str, help="simulation time (ns)", default="inp")
args = arg_parser.parse_args()
if '/' in os.getcwd():  # In order to make this work for both MAC and WINDOWS
    slash = '/'
elif '\\' in os.getcwd():
    slash = '\\'
else:
    raise Exception("Unable to determine directory.")
#temp_path_out = args.output.rstrip(slash) + slash + args.sim_time
#if not os.path.exists(temp_path_out):
#    os.makedirs(temp_path_out)
#print(os.getcwd())
#print(args.input.rstrip(slash))
#temp_path_in = args.input.rstrip(slash)
temp_path_in = os.getcwd()
if len(args.volt_list) > 0:
    with open(temp_path_in + slash + args.volt_list, "r") as fh:
        volt_list = [ i for i in fh.readlines()]
        #print(volt_list)
else:
    volt_list = None
para_list_raw = list(map(lambda x: tuple(x.split(":")), [i for i in args.para_list.split(";")]))
para_list = list()
for para, para_file in para_list_raw:
    with open(temp_path_in.rstrip(slash) + slash + para_file, "r") as fh:
        temp_list = [i.rstrip() for i in fh.readlines() if len(i) > 0]
    if temp_list[0].startswith("/"):
        content_list = list()
        sub_dir = temp_path_in+temp_list[0].rstrip(slash)+slash
        for fname in temp_list[1:]:
            with open(sub_dir+fname, "r") as fh:
                content_list.append(fh.read())
    else:
        content_list = temp_list
    para_list.append((para, content_list))
    print(para_list)
# Check if len of each content_list are the same
#lst = [len(i[1]) for i in para_list]
#if not lst[1:] == lst[:-1]:
#    raise Exception("Parameter lists should have the same number of items! Length of the items:", lst)
with open(temp_path_in+slash+"qsub.sh", "w") as fh:
    if volt_list is None:
        fh.writelines(["qsub "+str(i.strip().split(".")[0])+"V.pbs\n" for i in range(lst[0])])
    else:
        fh.writelines(["qsub " + i.strip().split(".")[0] + "V.pbs\n" for i in volt_list])
label_list = [i[0] for i in para_list]
para2replace = zip(*(i[1] for i in para_list))
for i, para_set in enumerate(para2replace):
    if volt_list is None:
        volt = str(i)
    else:
        volt = volt_list[i]
        volt = str(volt).strip().split(".")[0]
        #print(volt)
    fh_out = open(temp_path_in+slash+volt+"V.pbs", "w")
    time = args.sim_time
    with open(temp_path_in + slash + args.template, "r") as fh_temp:
        for in_line in fh_temp:
            # http://stackoverflow.com/questions/6116978/python-replace-multiple-strings
            rep = {"VOLT": volt, "TIME": time}  # define desired replacements here
            for label, para in zip(label_list, para_set):
                rep[label] = para
            # use these three lines to do the replacement
            rep = dict((re.escape(k), v) for k, v in rep.items())
            pattern = re.compile("|".join(rep.keys()))
            text = pattern.sub(lambda m: rep[re.escape(m.group(0))], in_line)
            fh_out.write(text)
    fh_out.close()

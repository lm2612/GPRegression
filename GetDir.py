import os

def GetDir():
        cwd = os.getcwd()
        print("Directory: {0}".format(cwd))
        return('/work/lm2612/data/AvgData/')
        if 'tmp/' in cwd:
           # on cx1
           datadir = '/work/lm2612/data/AvgData/'
        if ('cx1' in cwd) or ('/rds/' in cwd):
           # on cx1
           datadir = '/work/lm2612/data/AvgData/'
        elif ( '/home/d01/laman' in cwd ) or ('/scratch/jtmp/'in cwd ):
           # on Monsoon
           datadir = '/projects/ukca-imp/laman/AllFiles/AvgData/'
        elif '/Users/lm2612/Documents/' in cwd:
           # On Laura's Mac
           datadir = '/Users/lm2612/Documents/PhD/data/AvgData/'
        elif '/home/laura/' in cwd:
           datadir = '/home/laura/Documents/data/AvgData/'
        else:
            print('What PC are you on? Do not recognise from current directory so unable to define directory to read or write files. Please define manually.')
            exit()
        return(datadir)

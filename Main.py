import sys

from Modules.Visualization import VisualizationClass


def main():
    args = sys.argv

    # Arguments:
    # --h show help
    # --v visualize data

    print("Starting analysis with #{0} parameters:".format(len(args)))

    if(len(args) == 1):
        print("Show help")
    else:
        if(len(args) >= 2 and args[1] == '--v'):
            visualize = VisualizationClass()

            visualize.loadData()

            if (args[2].lower() == 'workingrawdata'):
                visualize.workingRawData()
            elif(args[2].lower() == 'agentstable'):
                visualize.agentsTable()
            elif(args[2].lower() == 'visualizeav'):
                visualize.visualizeAV()
            elif(args[2].lower() == 'visualizeagent'):
                visualize.visualizeAgent()

if __name__ == "__main__":
    main()
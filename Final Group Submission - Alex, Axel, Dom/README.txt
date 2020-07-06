Download the model and flight data here: https://drive.google.com/file/d/172zg0XsXpOOGNfm7Kn3oQgBZpjO83KPB/view?usp=sharing
Unzip the files and put the "flight-data" folder and "rf_model.joblib" in the same directory as "gui.py"

-----------------------
Running the GUI
-----------------------

Ensure SKLearn is installed

To run the GUI:
- Place the gui.py file and the rf_model.joblib files in the same directory
- Type "python3 gui.py" in the command line (can optionally use "python3 gui.py new_model.joblib" to try another model)
- Enter the information asked in the format specified by the script

-----------------------
Creating the Model
-----------------------

Ensure SKLearn, numpy, csv, and pandas are installed

To run the Model Creation:
- Place the create_models.py file into the same directory as the flight-data folder
- Ensure all flight data needed for analysis are in CSV format and in the folder flight-data
- Type "python3 create_models.py #OFYEARS AIRPORT_CODE" in the command line
	- REQUIRED Parameters:
		- #OFYEARS: Value between 1-10 inclusive. Species the number of flight data years to run on
	- NOT REQUIRED Parameters:
		- AIRPORT_CODE: 3 letter op-code ie. CLE, ATL, LAX. To use all airports use keyword ALL (default value = ALL)
	
- Examples:
	- "python3 create_models.py 2 ATL"
		- creates model using 2 years and only flights from the Atlanta airport
	- "python 3 create_models.py 1 ALL" OR "python3 create_models.py 1"
		- create model using 1 year and all flight data



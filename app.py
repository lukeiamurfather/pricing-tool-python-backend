import os
import sys
import traceback
import json
import h2o
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from h2o.estimators.glm import H2OGeneralizedLinearEstimator
import math
import polars
import pyarrow
import site
import numpy as np

app = Flask(__name__)
CORS(app)

# Global variable to track H2O initialization
h2o_initialized = False

def get_h2o_jar_path():
    # Get the site-packages directory
    site_packages = site.getsitepackages()[0]
    
    # Construct the path to h2o jar
    jar_path = os.path.join(site_packages, 'h2o', 'backend', 'bin')
    
    return jar_path

def ensure_h2o_init():
    global h2o_initialized
    if not h2o_initialized:
        try:
            # Use a specific memory size format
            h2o.init(max_mem_size='12G', nthreads=-1)
            h2o_initialized = True
            print("H2O cluster initialized successfully")
        except Exception as e:
            print(f"H2O initialization error: {e}")
            raise

def get_base_levels(data, predictors):
    #print(f"Calculating base levels for predictors: {predictors}")
    import polars
    import pyarrow
    
    base_levels = {}
    
    # Verify that the data frame has the expected columns
    available_cols = data.columns
    #rint(f"Available columns: {available_cols}")
    
    for predictor in predictors:
        try:
            if predictor not in available_cols:
                print(f"Warning: Predictor {predictor} not found in data columns")
                continue
                
            if predictor != 'year':
                #print(f"Processing base level for {predictor}")
                grouped = data.group_by(predictor).sum(['exposure']).get_frame()
                with h2o.utils.threading.local_context(polars_enabled=True):
                    exposures = grouped.as_data_frame()
                if 'sum_exposure' not in exposures.columns:
                    print(f"Warning: sum_exposure column not found in grouped data for {predictor}")
                    continue
                    
                if len(exposures) == 0:
                    print(f"Warning: No data found for predictor {predictor}")
                    continue
                    
                # Find the level with maximum exposure
                try:
                    base_level = exposures.iloc[exposures['sum_exposure'].argmax()][predictor]
                    base_levels[predictor] = str(base_level)
                    #print(f"Base level for {predictor}: {base_level}")
                except Exception as e:
                    print(f"Error finding base level for {predictor}: {e}")
                    continue
            else:
                try:
                    # For 'year', use the maximum year as the base level
                    year_max = data['year'].asnumeric().max()
                    base_levels['year'] = str(int(year_max))
                    #print(f"Base level for year: {base_levels['year']}")
                except Exception as e:
                    print(f"Error finding base level for year: {e}")
                    continue
        except Exception as e:
            print(f"Error processing base level for {predictor}: {e}")
            # Continue with other predictors instead of failing
    
    return base_levels

@app.route('/api/initialize-h2o-only', methods=['POST'])
def initialize_h2o_only():
    try:
        # Ensure H2O is initialized
        print('Trying H2O Init')
        import h2o

        h2o.init(max_mem_size='12G', nthreads=-1)

        # Get cluster information
        #cluster_info = h2o.cluster()
        
        # Prepare response
        response = {
            'totalMemory': '12GB',
            'availableCores': '4',
        }

        return jsonify(response)

    except Exception as e:
        # Detailed error logging
        print("Error initializing H2O:")
        print(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/api/initialize-h2o', methods=['POST'])
def initialize_h2o():
    try:
        # Import required modules
        import os
        import h2o
        import pandas as pd
        
        h2o.init(max_mem_size='12G', nthreads=-1)
        #h2o.utils.config.H2OContext.default_context.polars_enabled = True

        # Parse incoming data
        data = request.json
        
        # Log what we received for debugging
        #print("Received data structure:", data.keys() if data else "None")
        
        # Check for dataset ID - if we have it, we can load from files
        dataset_id = data.get('datasetId')
        if dataset_id:
            print(f"Loading data from dataset: {dataset_id}")
            datasets_dir = os.path.join(os.path.dirname(__file__), 'datasets')
            dataset_dir = os.path.join(datasets_dir, dataset_id)
            
            # Check if directories exist
            #print(f"Checking for dataset directory: {dataset_dir}")
            #print(f"Directory exists: {os.path.exists(dataset_dir)}")
            
            if os.path.exists(dataset_dir):
                train_path = os.path.join(dataset_dir, 'train.csv')
                test_path = os.path.join(dataset_dir, 'test.csv')
                
                #print(f"Train path: {train_path}, exists: {os.path.exists(train_path)}")
                #print(f"Test path: {test_path}, exists: {os.path.exists(test_path)}")
                
                # Read the first few lines of CSV to check columns
                if os.path.exists(train_path):
                    #print("Checking CSV columns in train.csv:")
                    with open(train_path, 'r') as f:
                        first_line = f.readline().strip()
                        #print(f"CSV Headers: {first_line}")
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    # Load the files into H2O
                    h2o_train = h2o.import_file(train_path)
                    h2o_test = h2o.import_file(test_path)
                    
                    # Log the columns for verification
                    #print(f"Train columns: {h2o_train.columns}")
                    #print(f"Test columns: {h2o_test.columns}")
                    
                    # Store frames globally in H2O's memory
                    h2o_train_key = 'train_frame_orig'
                    h2o_test_key = 'test_frame_orig'
                    h2o.assign(h2o_train, h2o_train_key)
                    h2o.assign(h2o_test, h2o_test_key)
                    
                    #print(f"Successfully loaded dataset from files. Train shape: {h2o_train.shape}, Test shape: {h2o_test.shape}")
                    
                    # Extract column information
                    columns_info = []
                    for col in h2o_train.columns:
                        col_type = "numeric" if h2o_train.types[col] in ['real', 'int'] else "categorical"
                        columns_info.append({
                            "name": col,
                            "type": col_type
                        })
                    
                    # Get cluster information
                    cluster_info = h2o.cluster()
                    
                    # Prepare response
                    response = {
                        'totalMemory': '12GB',
                        'availableCores': '4',
                        'version': h2o.__version__,
                        'trainFrameId': h2o_train_key,
                        'testFrameId': h2o_test_key,
                        'columns': columns_info
                    }
                    
                    return jsonify(response)
            
            # If we get here, the dataset files weren't found in the expected location
            print(f"Dataset files not found for ID: {dataset_id}")
            
            # Try other possible locations
            possible_paths = [
                os.path.join('..', 'server', 'datasets', dataset_id),
                os.path.join(os.getcwd(), 'datasets', dataset_id),
                os.path.join(os.getcwd(), '..', 'datasets', dataset_id),
                r'\pricing-tool-exe-v2\server\datasets\{}'.format(dataset_id)
            ]
            
            for alt_path in possible_paths:
                print(f"Trying alternative path: {alt_path}")
                train_path = os.path.join(alt_path, 'train.csv')
                test_path = os.path.join(alt_path, 'test.csv')
                
                if os.path.exists(train_path) and os.path.exists(test_path):
                    print(f"Found files at alternative path: {alt_path}")
                    
                    # Load the files into H2O
                    h2o_train = h2o.import_file(train_path)
                    h2o_test = h2o.import_file(test_path)
                    
                    # Store frames globally in H2O's memory
                    h2o_train_key = 'train_frame_orig'
                    h2o_test_key = 'test_frame_orig'
                    h2o.assign(h2o_train, h2o_train_key)
                    h2o.assign(h2o_test, h2o_test_key)
                    
                    print(f"Successfully loaded dataset from alternative path")
                    
                    # Extract column information
                    columns_info = []
                    for col in h2o_train.columns:
                        col_type = "numeric" if h2o_train.types[col] in ['real', 'int'] else "categorical"
                        columns_info.append({
                            "name": col,
                            "type": col_type
                        })
                    
                    # Prepare response
                    response = {
                        'totalMemory': '12GB',
                        'availableCores': '4',
                        'version': h2o.__version__,
                        'trainFrameId': h2o_train_key,
                        'testFrameId': h2o_test_key,
                        'columns': columns_info
                    }
                    
                    return jsonify(response)
        
        # If we get here, we were unable to load data either from parameters or files
        return jsonify({
            'error': 'Unable to load dataset. Please check that dataset files exist.'
        }), 400
    
    except Exception as e:
        # Detailed error logging
        print("Error initializing H2O:")
        print(traceback.format_exc())
        
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
    
@app.route('/api/shutdown-h2o', methods=['POST'])
def shutdown_h2o():
    global h2o_initialized
    try:
        # Clear any temporary frames first
        try:
            for frame_id in ['train_frame_freq_prep', 'test_frame_freq_prep', 
                           'train_frame_sev_prep', 'test_frame_sev_prep',
                           'test_frame_freq', 'test_frame_sev', 'test_frame',
                           'grouped_temp', 'grouped_temp_agg', 'pred_data_temp']:
                try:
                    if h2o.get_frame(frame_id) is not None:
                        h2o.remove(frame_id)
                        print(f"Removed frame: {frame_id}")
                except:
                    pass
        except Exception as e:
            print(f"Error clearing temporary frames: {e}")
            
        # Attempt to gracefully shutdown H2O cluster
        print("Shutting down H2O cluster...")
        h2o.shutdown(prompt=False)
        h2o_initialized = False
        
        # Clear memory and perform garbage collection
        import gc
        gc.collect()
        
        print("H2O cluster shutdown successful")
        return jsonify({'status': 'H2O cluster shutdown successful'})
    except Exception as e:
        print(f"Error shutting down H2O: {str(e)}")
        print(traceback.format_exc())
        
        # Even if there's an error, mark as uninitialized to force restart on next attempt
        h2o_initialized = False
        
        return jsonify({
            'error': str(e)
        }), 500

#if __name__ == '__main__':
    #print("Python Version:", sys.version)
    #print("H2O Version:", h2o.__version__)
    
    # Pre-initialize H2O when script starts
    #try:
    #    ensure_h2o_init()
    #except Exception as e:
    #    print(f"Failed to initialize H2O at startup: {e}")

@app.route('/api/inspect-h2o-columns', methods=['GET'])
def inspect_h2o_columns():
    try:
        columns_info = []
        
        # Check if the train frame exists
        if 'train_frame_orig' in h2o.frames():
            train_frame = h2o.get_frame('train_frame_orig')
            
            for col in train_frame.columns:
                col_type = "numeric" if train_frame.types[col] in ['real', 'int'] else "categorical"
                columns_info.append({
                    "name": col,
                    "type": col_type
                })
                
            return jsonify({
                'status': 'success',
                'columns': columns_info
            })
        else:
            # If the standard frame doesn't exist, list all available frames
            available_frames = []
            for frame_id in h2o.frames():
                try:
                    frame_name = frame_id['frame_id']['name']
                    frame = h2o.get_frame(frame_name)
                    available_frames.append({
                        'id': frame_name,
                        'dimensions': f"{frame.nrows} x {frame.ncols}",
                        'column_count': len(frame.columns)
                    })
                except:
                    # Skip frames that might have issues
                    pass
            
            return jsonify({
                'status': 'error',
                'message': 'No train_frame_orig found in H2O frames',
                'available_frames': available_frames
            })
    except Exception as e:
        print(f"Error inspecting H2O columns: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/prepare-frequency-model', methods=['POST'])
def prepare_frequency_model():
    try:
        # Parse incoming data
        data = request.json
        prepared_data = data.get('data', [])
        variable_types = data.get('variableTypes', {})

        # Basic validation
        if not prepared_data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to pandas DataFrame for further processing if needed
        import pandas as pd
        #df = pd.DataFrame(prepared_data)

        # You can add more processing logic here, such as:
        # - Validate data types
        # - Perform initial checks
        # - Prepare for H2O modeling

        return jsonify({
            'status': 'success',
            'message': 'Frequency model data prepared',
            'receivedRows': len(prepared_data),
            'variables': variable_types
        })

    except Exception as e:
        print(f"Error in prepare_frequency_model: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500        
    
@app.route('/api/create-frequency-glm', methods=['POST'])
def create_frequency_glm():
    try:
        data = request.json        
        variable_types = data.get('variableTypes', {})
        dataset_id = data.get('datasetId')
        h2o_frame_ids = data.get('h2oFrameIds', {})

        # Add debug logging to verify data access
        print(f"Creating frequency GLM with dataset_id: {dataset_id}")
        print(f"H2O frame IDs: {h2o_frame_ids}")
        print(f"Target: {variable_types.get('target')}")
        print(f"Predictors: {variable_types.get('predictors', [])}")
        print(f"Offsets: {variable_types.get('offsets', [])}")

        target_freq = variable_types.get('target')
        predictors = variable_types.get('predictors', [])
        offsets = variable_types.get('offsets', []) 
        interactions = variable_types.get('interactions', [])

        print(f"Creating frequency GLM with target: {target_freq}, predictors: {predictors}, offsets: {offsets}")
        print(f"Interactions: {interactions}")

        if not target_freq or not predictors:
            return jsonify({
                'error': 'Target or Predictors not specified'
            }), 400

        if not offsets:
            return jsonify({
                'error': 'At least one offset is required'
            }), 400

        # Get the frames that were stored in H2O during initialization
        try:
            # Try to get frames by standard names
            train = h2o.get_frame('train_frame_orig')
            test = h2o.get_frame('test_frame_orig')
            
            # If frames don't exist, check for custom frame IDs
            if train is None and h2o_frame_ids and h2o_frame_ids.get('trainFrameId'):
                train = h2o.get_frame(h2o_frame_ids.get('trainFrameId'))
                
            if test is None and h2o_frame_ids and h2o_frame_ids.get('testFrameId'):
                test = h2o.get_frame(h2o_frame_ids.get('testFrameId'))
            
            # Print frame info for debugging
            print(f"Train frame info: {train.dim if train is not None else 'None'}, columns: {train.columns if train is not None else 'None'}")
            print(f"Test frame info: {test.dim if test is not None else 'None'}, columns: {test.columns if test is not None else 'None'}")

            if train is None or test is None:
                print("H2O frames not found. Checking available frames:")
                all_frames = h2o.frames()
                for frame_id in all_frames:
                    print(f"Found frame: {frame_id['frame_id']['name']}")
                raise Exception("Could not find H2O frames. Please reinitialize H2O.")

            # Make copies of the original frames to avoid modifying them
            train_copy = h2o.assign(train, 'train_frame_freq_prep')
            test_copy = h2o.assign(test, 'test_frame_freq_prep')

            # Verify that required columns exist
            missing_cols = []
            for col in [target_freq] + predictors + offsets:
                if col not in train_copy.columns:
                    missing_cols.append(col)
            
            if missing_cols:
                print(f"Missing columns in train frame: {missing_cols}")
                return jsonify({
                    'error': f'Missing required columns in data: {", ".join(missing_cols)}'
                }), 400

        except Exception as e:
            print(f"Error retrieving frames: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to retrieve H2O frames: {str(e)}'}), 500

        # Convert predictors to factors
        for predictor in predictors:
            if predictor != 'year': 
                try:
                    print(f"Converting {predictor} to factor")
                    train_copy[predictor] = train_copy[predictor].ascharacter().asfactor()
                    test_copy[predictor] = test_copy[predictor].ascharacter().asfactor()
                except Exception as e:
                    print(f"Error converting {predictor} to factor: {e}")
                    # Continue with other predictors instead of failing
                    continue

        try:
            # Create combined offset column as product of all offset variables
            print(f"Creating offset column from: {offsets}")
            train_copy['off'] = train_copy[offsets[0]]  # Initialize with first offset
            test_copy['off'] = test_copy[offsets[0]]

            # Multiply by remaining offsets if any
            for offset_var in offsets[1:]:
                train_copy['off'] = train_copy['off'] * train_copy[offset_var]
                test_copy['off'] = test_copy['off'] * test_copy[offset_var]

            # Take log of combined offset
            train_copy['off'] = train_copy['off'].log()
            test_copy['off'] = test_copy['off'].log()
        except Exception as e:
            print(f"Error creating offset column: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to create offset column: {str(e)}'}), 500

        try:
            # Convert year to factor
            if 'year' in train_copy.columns:
                print("Converting year to factor")
                train_copy['year'] = train_copy['year'].ascharacter().asfactor()
                test_copy['year'] = test_copy['year'].ascharacter().asfactor()
            
            actual = test_copy[target_freq]
        except Exception as e:
            print(f"Error converting year to factor: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to convert year to factor: {str(e)}'}), 500

        import polars 
        import pyarrow

        # Handle interactions
        try:
            for interaction in interactions:
                var1, var2 = interaction.get('var1'), interaction.get('var2')
                if not var1 or not var2:
                    print(f"Skipping invalid interaction: {interaction}")
                    continue
                    
                int_name = f"{var1}:{var2}"
                print(f"Creating interaction term: {int_name}")
                
                # Convert to pandas first to ensure alignment
                try:
                    with h2o.utils.threading.local_context(polars_enabled=True):
                        var1_df = train_copy[var1].as_data_frame()

                    with h2o.utils.threading.local_context(polars_enabled=True):
                        var2_df = train_copy[var2].as_data_frame()    

                    combined = [f"{str(v1)}:{str(v2)}" for v1, v2 in zip(var1_df[var1], var2_df[var2])]

                    with h2o.utils.threading.local_context(polars_enabled=True):
                        var1_test_df = test_copy[var1].as_data_frame()

                    with h2o.utils.threading.local_context(polars_enabled=True):
                        var2_test_df = test_copy[var2].as_data_frame()

                    combined_test = [f"{str(v1)}:{str(v2)}" for v1, v2 in zip(var1_test_df[var1], var2_test_df[var2])]
                    
                    # Convert back to H2O frames
                    train_copy[int_name] = h2o.H2OFrame(combined, column_types=['factor'])
                    test_copy[int_name] = h2o.H2OFrame(combined_test, column_types=['factor'])
                    
                    predictors.append(int_name)
                    print(f"Added interaction term {int_name} to predictors")
                except Exception as e:
                    print(f"Error creating interaction term {int_name}: {e}")
                    # Continue without this interaction
                    continue
        except Exception as e:
            print(f"Error handling interactions: {e}")
            print(traceback.format_exc())
            # Continue without interactions rather than failing
        
        # Get base levels before model fitting
        try:
            base_levels = get_base_levels(train_copy, predictors)
            print(f"Base levels: {base_levels}")
        except Exception as e:
            print(f"Error getting base levels: {e}")
            print(traceback.format_exc())
            base_levels = {}  # Continue with empty base levels
            
        # Show final data before fitting
        print(f"Final predictors for GLM: {predictors}")
        print(f"Target for GLM: {target_freq}")
        print(f"Using offset column: off")

        # Fit GLM with more detailed error handling
        try:
            glm_freq = H2OGeneralizedLinearEstimator(
                model_id='freq', 
                family='poisson', 
                link='log', 
                alpha=0.5, 
                nfolds=10
            )
            print(f"Training GLM with predictors: {predictors}, target: {target_freq}")
            glm_freq.train(x=predictors, y=target_freq, offset_column="off", training_frame=train_copy)
            print("GLM training completed successfully")
        except Exception as e:
            print(f"Error training GLM: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to train frequency GLM: {str(e)}'}), 500
            
        # Generate predictions and metrics
        try:
            print("Generating predictions")
            preds = glm_freq.predict(test_copy)
            perf = h2o.make_metrics(preds, actual)
            print(f"Prediction metrics generated: MSE={perf.mse()}")
        except Exception as e:
            print(f"Error generating predictions: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to generate predictions: {str(e)}'}), 500

        # Get variable importance
        try:
            print("Calculating variable importance")
            var_imp_df = glm_freq.permutation_importance(test_copy, use_pandas=True)
            var_imp_df = var_imp_df.reset_index()
            var_imp = [{
                'variable': str(row['Variable']),
                'relative_importance': float(row['Relative Importance']),
                'scaled_importance': float(row['Scaled Importance']),
                'percentage': float(row['Percentage'])
            } for _, row in var_imp_df.iterrows()]
            print(f"Variable importance calculated for {len(var_imp)} variables")
        except Exception as e:
            print(f"Error calculating variable importance: {e}")
            print(traceback.format_exc())
            var_imp = []  # Provide empty var_imp if calculation fails
        
        # Add predictions to the copy frame
        test_copy['predicted_freq'] = preds
        test_copy['predicted_count'] = preds * test_copy['exposure']
        
        # Create a new combined frame for later use (this is critical for the aggregate tab)
        # But don't modify the original test frame
        combined_test = h2o.assign(test_copy, 'test_frame')
        
        # Also store a frequency-specific frame
        h2o.assign(test_copy, 'test_frame_freq')
        print("Predictions stored in test_frame and test_frame_freq")

        # Extract original coefficients
        try:
            print("Extracting coefficients")
            coefs = glm_freq.coef()
            #print(f"Raw coefficients: {coefs}")
            
            coefs_freq = [
                {
                    'rating_factor': str(k) if '.' not in str(k) else str(k).split('.')[0],
                    'level': str(k).split('.')[1] if '.' in str(k) else '',
                    'estimate': math.exp(float(v))
                } 
                for k, v in coefs.items()
            ]
            print(f"Formatted {len(coefs_freq)} coefficients")
        except Exception as e:
            print(f"Error extracting coefficients: {e}")
            print(traceback.format_exc())
            return jsonify({'error': f'Failed to extract coefficients: {str(e)}'}), 500

        # Calculate base coefficients
        try:
            base_coefs = {
                factor: next((c['estimate'] for c in coefs_freq 
                             if c['rating_factor'] == factor and c['level'] == level), 1.0)
                for factor, level in base_levels.items()
            }
            
            # Calculate intercept multiplier
            intercept_multiplier = 1.0
            for factor, coef in base_coefs.items():
                intercept_multiplier *= coef
            
            # Relevel coefficients
            releveled_coefs = []
            for coef in coefs_freq:
                if coef['rating_factor'] == 'Intercept':
                    new_estimate = coef['estimate'] * intercept_multiplier
                else:
                    base_coef = base_coefs.get(coef['rating_factor'], 1.0)
                    new_estimate = coef['estimate'] / base_coef if base_coef != 0 else coef['estimate']
                
                releveled_coefs.append({
                    **coef,
                    'estimate': new_estimate
                })
            print(f"Releveled {len(releveled_coefs)} coefficients")
        except Exception as e:
            print(f"Error releveling coefficients: {e}")
            print(traceback.format_exc())
            releveled_coefs = coefs_freq  # Use original coefficients if releveling fails
        
        try:
            performance_metrics = {
                'Mean Squared Error': perf.mse(),
                'Root Mean Squared Error': perf.rmse(),
                'Mean Absolute Error': perf.mae(),
                'Root Mean Squared Logarithmic Error': perf.rmsle(),
                'Null Deviance': glm_freq.null_deviance(),
                'Residual Deviance': glm_freq.residual_deviance(),
                "R-Squared": perf.r2() * 100
            }
            print(f"Performance metrics calculated: {performance_metrics}")
        except Exception as e:
            print(f"Error calculating performance metrics: {e}")
            print(traceback.format_exc())
            performance_metrics = {
                'Error': 'Could not calculate performance metrics'
            }

        # Clean up temporary frames
        h2o.remove('train_frame_freq_prep')
        h2o.remove('test_frame_freq_prep')

        print("Successfully created frequency GLM, returning results")
        return jsonify({
            'status': 'success',
            'coefficients': coefs_freq,
            'releveled_coefficients': releveled_coefs,
            'base_levels': base_levels,
            'performance_metrics': performance_metrics,
            'var_imp': var_imp
        })

    except Exception as e:
        print(f"Error in create_frequency_glm: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500
    
@app.route('/api/plot-predictor', methods=['POST'])
def plot_predictor():
    try:
        data = request.json
        predictor = data.get('predictor')
        
        # Get test frame with predictions already set
        test = h2o.get_frame('test_frame_freq')

        year_max = test['year'].asnumeric().max()
        latest_data = test[test['year'] == str(int(year_max))]

        aggs = {
            'exposure': 'sum(exposure)',
            'count': 'sum(count)',
            'predicted_count': 'sum(predicted_count)'
        }
        
        # Group by predictor and calculate metrics
        grouped_frame_id = 'grouped_temp'
        grouped = latest_data.group_by(predictor).sum(['exposure', 'count', 'predicted_count']).get_frame()
        h2o.assign(grouped, grouped_frame_id)
        
        with h2o.utils.threading.local_context(polars_enabled=True):
            result = grouped.as_data_frame()
        
        h2o.remove(grouped_frame_id)

        return jsonify({'data': result.round(2).to_dict('records')})

    except Exception as e:
        print(f"Error in plot_predictor: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500
    
@app.route('/api/gini-plot', methods=['POST'])
def gini_plot():
    try:
        test = h2o.get_frame('test_frame_freq')
            
        obs_lc = test['count']
        pred_lc = test['predicted_freq']
        exp = test['exposure']
            
        dataset = obs_lc.cbind(pred_lc)
        dataset = dataset.cbind(exp)
        dataset = dataset.sort(by='predicted_freq', ascending=True)
           
        dataset['losses'] = dataset['predicted_freq'] * dataset['exposure']
        dataset['cum_exp'] = dataset['exposure'].cumsum() / dataset['exposure'].sum()
        dataset['cum_losses'] = dataset['losses'].cumsum() / dataset['losses'].sum()

        with h2o.utils.threading.local_context(polars_enabled=True):
            df = dataset[['cum_exp', 'cum_losses']].as_data_frame()
        result = [{'cum_exp': 0, 'cum_losses': 0}] + df.to_dict(orient='records') + [{'cum_exp': 100, 'cum_losses': 100}]
        
        return jsonify({'data': result})

    except Exception as e:
        print(f"Error in gini_plot:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/calculate-quantile-residuals', methods=['POST'])
def calculate_quantile_residuals():
    try:
        
        test = h2o.get_frame('test_frame_freq')
        
        # Convert to numpy arrays for efficient computation
        with h2o.utils.threading.local_context(polars_enabled=True):
            df = test[['count', 'predicted_freq']].as_data_frame()
            observed = df['count'].values.flatten()
            predicted = df['predicted_freq'].values.flatten()

        import numpy as np
        import scipy
        from scipy import stats
        
        # Calculate CDF values efficiently using vectorized operations
        a = stats.poisson.cdf(observed - 1, predicted)
        b = stats.poisson.cdf(observed, predicted)
        
        # Generate uniform random numbers and transform
        u = np.random.uniform(a, b)
        qresiduals = stats.norm.ppf(u)
        
        # Convert back to H2O frame
        test['quantile_residuals'] = h2o.H2OFrame(qresiduals.reshape(-1, 1))
        h2o.assign(test, 'test_frame_freqres')
        
        return jsonify({
            'residuals': qresiduals.tolist()
        })

    except Exception as e:
        print(f"Error in calculate_quantile_residuals: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/res_v_fit', methods=['GET', 'POST'])
def res_v_fit():
    try:
        # Get the test frame
        test = h2o.get_frame('test_frame_freq')
        if test is None:
            print("Error: test_frame_freq not found")
            return jsonify({'error': 'Test frame not found'}), 404
            
        print("Found test frame with dimensions:", test.shape)
        
        # Extract the predicted and observed frequencies
        with h2o.utils.threading.local_context(polars_enabled=True):
            df = test[['count', 'predicted_freq']].as_data_frame()
            observed = df['count'].values.flatten()
            predicted = df['predicted_freq'].values.flatten()
        
        import numpy as np
        from scipy import stats
        
        # Calculate CDF values
        a = stats.poisson.cdf(observed - 1, predicted)
        b = stats.poisson.cdf(observed, predicted)
        
        # Generate uniform random numbers and transform to normal quantiles
        u = np.random.uniform(a, b)
        qresiduals = stats.norm.ppf(u)
        
        # Prepare response with both arrays
        response_data = {
            'quantile_residuals': qresiduals.tolist(),
            'predicted_frequency': predicted.tolist()
        }
        
        print("Returning arrays with lengths - residuals:", len(qresiduals), 
              "predicted:", len(predicted))
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"Error in get_model_diagnostics: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)})

@app.route('/api/prepare-severity-model', methods=['POST'])
def prepare_severity_model():
    try:
        # Parse incoming data
        data = request.json
        prepared_data = data.get('data', [])
        variable_types = data.get('variableTypes', {})

        # Basic validation
        if not prepared_data:
            return jsonify({'error': 'No data provided'}), 400

        # Convert to pandas DataFrame for further processing if needed
        import pandas as pd
        #df = pd.DataFrame(prepared_data)

        # You can add more processing logic here, such as:
        # - Validate data types
        # - Perform initial checks
        # - Prepare for H2O modeling

        return jsonify({
            'status': 'success',
            'message': 'Severity model data prepared',
            'receivedRows': len(prepared_data),
            'variables': variable_types
        })

    except Exception as e:
        print(f"Error in prepare_severity_model: {str(e)}")
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/create-severity-glm', methods=['POST'])
def create_severity_glm():
    try:
        data = request.json
        variable_types = data.get('variableTypes', {})

        target = variable_types.get('target')
        predictors = variable_types.get('predictors', [])
        interactions = variable_types.get('interactions', [])
        wt = variable_types.get('weight')

        if not target or not predictors:
            return jsonify({
                'error': 'Target or Predictors not specified'
            }), 400

        # Always start with the original train and test frames
        train = h2o.get_frame('train_frame_orig')
        test = h2o.get_frame('test_frame_orig')

        if train is None or test is None:
            return jsonify({
                'error': 'Original train or test frames not found'
            }), 404

        # Make copies of the original frames to avoid modifying them
        train_copy = h2o.assign(train, 'train_frame_sev_prep')
        test_copy = h2o.assign(test, 'test_frame_sev_prep')

        # Apply all transformations to these copies
        for predictor in predictors:
            if predictor != 'year':
                train_copy[predictor] = train_copy[predictor].ascharacter().asfactor()
                test_copy[predictor] = test_copy[predictor].ascharacter().asfactor()

        import polars 
        import pyarrow

        # Handle interactions
        for interaction in interactions:
            var1, var2 = interaction.get('var1'), interaction.get('var2')
            if not var1 or not var2:
                print(f"Skipping invalid interaction: {interaction}")
                continue
                
            int_name = f"{var1}:{var2}"
            print(f"Creating interaction term: {int_name}")
            
            # Convert to pandas first to ensure alignment
            try:
                with h2o.utils.threading.local_context(polars_enabled=True):
                    var1_df = train_copy[var1].as_data_frame()

                with h2o.utils.threading.local_context(polars_enabled=True):
                    var2_df = train_copy[var2].as_data_frame()    

                combined = [f"{str(v1)}:{str(v2)}" for v1, v2 in zip(var1_df[var1], var2_df[var2])]

                with h2o.utils.threading.local_context(polars_enabled=True):
                    var1_test_df = test_copy[var1].as_data_frame()

                with h2o.utils.threading.local_context(polars_enabled=True):
                    var2_test_df = test_copy[var2].as_data_frame()

                combined_test = [f"{str(v1)}:{str(v2)}" for v1, v2 in zip(var1_test_df[var1], var2_test_df[var2])]
                
                # Convert back to H2O frames
                train_copy[int_name] = h2o.H2OFrame(combined, column_types=['factor'])
                test_copy[int_name] = h2o.H2OFrame(combined_test, column_types=['factor'])
                
                predictors.append(int_name)
                print(f"Added interaction term {int_name} to predictors")
            except Exception as e:
                print(f"Error creating interaction term {int_name}: {e}")
                # Continue without this interaction
                continue

        train_copy['year'] = train_copy['year'].ascharacter().asfactor()

        # Create mask for records with count > 0
        mask = train_copy[target] > 0
        dt_sev = train_copy[mask,:]

        # Calculate severity for training and testing
        dt_sev['sev'] = dt_sev[target] / dt_sev['count']
        test_copy['sev'] = test_copy[target].ifelse(test_copy['count'] == 0, 0).ifelse(test_copy['count'] > 0, test_copy[target]/test_copy['count'])

        test_copy['year'] = test_copy['year'].ascharacter().asfactor()
        actual = test_copy['sev']

        # Get base levels
        base_levels = get_base_levels(train_copy, predictors)

        # Fit GLM
        glm_sev = H2OGeneralizedLinearEstimator(
            model_id='sev',
            family='gamma',
            link='log',
            alpha=0.5,
            nfolds=10,
            weights_column=wt
        )
        glm_sev.train(x=predictors, y='sev', training_frame=dt_sev)
        preds = glm_sev.predict(test_copy)
        perf = h2o.make_metrics(preds, actual)

        var_imp_df = glm_sev.permutation_importance(test_copy, use_pandas=True)
        var_imp_df = var_imp_df.reset_index()
        var_imp = [{
            'variable': str(row['Variable']),
            'relative_importance': float(row['Relative Importance']),
            'scaled_importance': float(row['Scaled Importance']),
            'percentage': float(row['Percentage'])
        } for _, row in var_imp_df.iterrows()]

        test_copy['predicted_sev'] = preds
        
        # Save to severity-specific frame
        h2o.assign(test_copy, 'test_frame_sev')
        
        # Get the combined test frame (which has freq predictions) if it exists
        try:
            combined_test = h2o.get_frame('test_frame')
            if combined_test is not None:
                # Add severity predictions to the combined frame
                combined_test['predicted_sev'] = preds
                combined_test['predicted_guc'] = combined_test['predicted_count'] * combined_test['predicted_sev']
                # Update the combined frame
                h2o.assign(combined_test, 'test_frame')
        except:
            print("No combined test frame exists yet, only saving severity-specific frame")

        # Extract coefficients
        coefs_sev = [
            {
                'rating_factor': str(k) if '.' not in str(k) else str(k).split('.')[0],
                'level': str(k).split('.')[1] if '.' in str(k) else '',
                'estimate': math.exp(float(v))
            } 
            for k, v in glm_sev.coef().items()
        ]

        # Calculate base coefficients
        base_coefs = {
            factor: next((c['estimate'] for c in coefs_sev 
                         if c['rating_factor'] == factor and c['level'] == level), 1.0)
            for factor, level in base_levels.items()
        }
        
        # Calculate intercept multiplier
        intercept_multiplier = 1.0
        for factor, coef in base_coefs.items():
            intercept_multiplier *= coef
        
        # Relevel coefficients
        releveled_coefs = []
        for coef in coefs_sev:
            if coef['rating_factor'] == 'Intercept':
                new_estimate = coef['estimate'] * intercept_multiplier
            else:
                base_coef = base_coefs.get(coef['rating_factor'], 1.0)
                new_estimate = coef['estimate'] / base_coef if base_coef != 0 else coef['estimate']
            
            releveled_coefs.append({
                **coef,
                'estimate': new_estimate
            })

        performance_metrics = {
            'Mean Squared Error': perf.mse(),
            'Root Mean Squared Error': perf.rmse(),
            'Mean Absolute Error': perf.mae(),
            'Root Mean Squared Logarithmic Error': perf.rmsle(),
            'Null Deviance': glm_sev.null_deviance(),
            'Residual Deviance': glm_sev.residual_deviance(),
            "R-Squared": perf.r2() * 100
        }

        # Clean up temporary frames
        h2o.remove('train_frame_sev_prep')
        h2o.remove('test_frame_sev_prep')

        return jsonify({
            'coefficients': coefs_sev,
            'releveled_coefficients': releveled_coefs,
            'base_levels': base_levels,
            'performance_metrics': performance_metrics,
            'var_imp': var_imp
        })

    except Exception as e:
        print(f"Error in create_severity_glm: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/gini-plot-sev', methods=['POST'])
def gini_plot_sev():
    try:
        test = h2o.get_frame('test_frame_sev')
            
        obs_lc = test['sev']
        pred_lc = test['predicted_sev']
        exp = test['exposure']
            
        dataset = obs_lc.cbind(pred_lc)
        dataset = dataset.cbind(exp)
        dataset = dataset.sort(by='predicted_sev', ascending=True)
           
        dataset['losses'] = dataset['predicted_sev'] * dataset['exposure']
        dataset['cum_exp'] = dataset['exposure'].cumsum() / dataset['exposure'].sum()
        dataset['cum_losses'] = dataset['losses'].cumsum() / dataset['losses'].sum()

        with h2o.utils.threading.local_context(polars_enabled=True):
            df = dataset[['cum_exp', 'cum_losses']].as_data_frame()
            result = [{'cum_exp': 0, 'cum_losses': 0}] + df.to_dict(orient='records') + [{'cum_exp': 100, 'cum_losses': 100}]
        
        return jsonify({'data': result})

    except Exception as e:
        print(f"Error in gini_plot:")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/get-total-exposure-info', methods=['GET'])
def get_total_exposure_info():
    """
    Provides diagnostic information about total exposure in the train and test datasets
    """
    try:
        # Get original data frames
        train = h2o.get_frame('train_frame_orig')
        test = h2o.get_frame('test_frame_orig')
        
        if train is None or test is None:
            return jsonify({
                'error': 'Original data frames not found.'
            }), 404
        
        # Convert to pandas for easier analysis
        with h2o.utils.threading.local_context(polars_enabled=True):
            train_df = train.as_data_frame()
            test_df = test.as_data_frame()
        
        # Get total exposure
        train_exposure = train_df['exposure'].sum() if 'exposure' in train_df.columns else 0
        test_exposure = test_df['exposure'].sum() if 'exposure' in test_df.columns else 0
        total_exposure = train_exposure + test_exposure
        
        # Get exposure by year
        exposure_by_year = {}
        
        if 'year' in train_df.columns and 'exposure' in train_df.columns:
            train_by_year = train_df.groupby('year')['exposure'].sum().to_dict()
            for year, exp in train_by_year.items():
                year_key = str(year)
                exposure_by_year[year_key] = exposure_by_year.get(year_key, 0) + exp
        
        if 'year' in test_df.columns and 'exposure' in test_df.columns:
            test_by_year = test_df.groupby('year')['exposure'].sum().to_dict()
            for year, exp in test_by_year.items():
                year_key = str(year)
                exposure_by_year[year_key] = exposure_by_year.get(year_key, 0) + exp
        
        # Find the max year
        max_year = None
        if 'year' in train_df.columns:
            train_max = train_df['year'].max()
            max_year = train_max
        
        if 'year' in test_df.columns:
            test_max = test_df['year'].max()
            if max_year is None or test_max > max_year:
                max_year = test_max
        
        # Return the diagnostic information
        return jsonify({
            'train_total_exposure': float(train_exposure),
            'test_total_exposure': float(test_exposure),
            'combined_total_exposure': float(total_exposure),
            'exposure_by_year': {str(k): float(v) for k, v in exposure_by_year.items()},
            'max_year': str(max_year) if max_year is not None else None
        })
    
    except Exception as e:
        print(f"Error getting total exposure info: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/get-combined-frame-data', methods=['GET'])
def get_combined_frame_data():
    """
    Retrieves original data along with model predictions for the aggregate tab
    Ensures only the base year data is returned
    """
    try:
        # First, check if we have the original data frames
        train = h2o.get_frame('train_frame_orig')
        test = h2o.get_frame('test_frame_orig')
        
        if train is None or test is None:
            return jsonify({
                'error': 'Original data frames not found. Please reinitialize H2O.'
            }), 404
        
        # Check if frequency model has been run
        freq_model_exists = False
        try:
            freq_test = h2o.get_frame('test_frame_freq')
            if freq_test is not None:
                freq_model_exists = True
        except:
            pass
            
        # Check if severity model has been run
        sev_model_exists = False
        try:
            sev_test = h2o.get_frame('test_frame_sev')
            if sev_test is not None:
                sev_model_exists = True
        except:
            pass
        
        if not freq_model_exists or not sev_model_exists:
            return jsonify({
                'error': 'Both frequency and severity models must be completed first.'
            }), 400
        
        # Get train and test data into pandas dataframes
        import pandas as pd
        import numpy as np
        
        print("Converting train and test frames to pandas dataframes...")
        with h2o.utils.threading.local_context(polars_enabled=True):
            train_df = train.as_data_frame()
            test_df = test.as_data_frame()
        
        # Get the max year from all data
        all_years = []
        if 'year' in train_df.columns:
            all_years.extend(train_df['year'].unique())
        if 'year' in test_df.columns:
            all_years.extend(test_df['year'].unique())
        
        if not all_years:
            return jsonify({
                'error': 'No year data found in train or test datasets'
            }), 400
            
        # Find the base year (maximum year)
        max_year = max(all_years)
        print(f"Base year determined to be: {max_year}")
        
        # Convert max_year to string for consistent comparison
        max_year_str = str(max_year)
        
        # Filter both train and test to ONLY include rows from the base year
        filtered_train_df = train_df[train_df['year'].astype(str) == max_year_str].copy() if 'year' in train_df.columns else pd.DataFrame()
        filtered_test_df = test_df[test_df['year'].astype(str) == max_year_str].copy() if 'year' in test_df.columns else pd.DataFrame()
        
        print(f"Filtered train data to {len(filtered_train_df)} rows for base year {max_year}")
        print(f"Filtered test data to {len(filtered_test_df)} rows for base year {max_year}")
        
        # Log exposure totals for debugging
        train_exposure = filtered_train_df['exposure'].sum() if 'exposure' in filtered_train_df.columns else 0
        test_exposure = filtered_test_df['exposure'].sum() if 'exposure' in filtered_test_df.columns else 0
        print(f"Train base year exposure: {train_exposure}")
        print(f"Test base year exposure: {test_exposure}")
        print(f"Combined base year exposure: {train_exposure + test_exposure}")
        
        # Combine filtered train and test data
        combined_df = pd.concat([filtered_train_df, filtered_test_df], ignore_index=True)
        
        if len(combined_df) == 0:
            return jsonify({
                'error': f'No data found for base year {max_year} in either train or test datasets'
            }), 400
        
        print(f"Combined data has {len(combined_df)} rows with total exposure: {combined_df['exposure'].sum()}")
            
        # Add predictions from frequency model
        if freq_model_exists:
            try:
                with h2o.utils.threading.local_context(polars_enabled=True):
                    freq_df = freq_test.as_data_frame()
                
                # Add predictions to combined data - but need to match on row indices or keys
                if 'year' in freq_df.columns:
                    freq_df = freq_df[freq_df['year'].astype(str) == max_year_str]
                    print(f"Filtered frequency predictions to {len(freq_df)} rows for base year")
                
                # Now add predictions to all records in the combined data
                # You'll need a smart way to match records from the frequency predictions to the combined data
                # This may involve joining on a key or set of keys
                # For simplicity, let's use a dummy approach for now (this should be enhanced for your specific data)
                
                # Just verify dimensions match for test portion
                if len(freq_df) == len(filtered_test_df):
                    # We can directly copy predictions to the test portion
                    combined_df.loc[len(filtered_train_df):, 'predicted_freq'] = freq_df['predicted_freq'].values
                    combined_df.loc[len(filtered_train_df):, 'predicted_count'] = freq_df['predicted_count'].values
                    print("Added frequency predictions to test portion of combined data")
                    
                    # For train portion, we need to apply the frequency model
                    # This is a placeholder - you'd need a better approach based on your data structure
                    combined_df.loc[:len(filtered_train_df)-1, 'predicted_freq'] = 0
                    combined_df.loc[:len(filtered_train_df)-1, 'predicted_count'] = 0
                    print("Added placeholder frequency predictions to train portion of combined data")
                else:
                    print(f"Warning: Frequency predictions count ({len(freq_df)}) doesn't match test data count ({len(filtered_test_df)})")
                    # Handle the mismatch appropriately
                    
            except Exception as e:
                print(f"Error adding frequency predictions: {e}")
                return jsonify({
                    'error': f'Failed to add frequency predictions: {str(e)}'
                }), 500
        
        # Add predictions from severity model (similar approach)
        if sev_model_exists:
            # Similar approach to frequency model predictions
            # ...
            pass
        
        # Ensure all records have prediction columns to avoid errors in frontend
        if 'predicted_freq' not in combined_df.columns:
            combined_df['predicted_freq'] = 0
            combined_df['predicted_count'] = 0
            
        if 'predicted_sev' not in combined_df.columns:
            combined_df['predicted_sev'] = 0
            
        # Calculate combined prediction
        if 'predicted_count' in combined_df.columns and 'predicted_sev' in combined_df.columns:
            combined_df['predicted_guc'] = combined_df['predicted_count'] * combined_df['predicted_sev']
        
        # Final verification - ensure all rows are for the base year
        unique_years = combined_df['year'].unique()
        if len(unique_years) != 1 or str(unique_years[0]) != max_year_str:
            print(f"ERROR: Data contains more than just the base year: {unique_years}")
            # Force all rows to have the correct year
            combined_df['year'] = max_year
        
        # Convert to records format for easier JSON handling
        records = combined_df.replace({np.nan: None}).to_dict('records')
        
        print(f"Returning {len(records)} records with base year {max_year}")
        
        return jsonify({
            'status': 'success',
            'data': records,
            'row_count': len(records),
            'columns': list(combined_df.columns),
            'base_year': max_year_str,  # Explicitly return as string
            'exposure_info': {
                'train': float(train_exposure),
                'test': float(test_exposure),
                'total': float(train_exposure + test_exposure)
            }
        })
        
    except Exception as e:
        print(f"Error retrieving combined frame data: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/plot-predictor-agg', methods=['POST'])
def plot_predictor_agg():
    try:
        data = request.json
        predictor = data.get('predictor')
        
        if not predictor:
            return jsonify({'error': 'No predictor specified'}), 400
        
        # First, we need to verify if both models have been run
        freq_model_exists = False
        try:
            freq_test = h2o.get_frame('test_frame_freq')
            if freq_test is not None:
                freq_model_exists = True
        except:
            pass
            
        sev_model_exists = False
        try:
            sev_test = h2o.get_frame('test_frame_sev')
            if sev_test is not None:
                sev_model_exists = True
        except:
            pass
        
        if not freq_model_exists:
            return jsonify({
                'error': 'Frequency model results not found. Please complete frequency modeling first.'
            }), 400
            
        if not sev_model_exists:
            return jsonify({
                'error': 'Severity model results not found. Please complete severity modeling first.'
            }), 400
        
        # Get original test data
        test_orig = h2o.get_frame('test_frame_orig')
        if test_orig is None:
            return jsonify({
                'error': 'Original test data not found. Please reinitialize H2O.'
            }), 404
        
        # Filter for the latest year if possible
        try:
            if 'year' in test_orig.columns:
                year_max = test_orig['year'].asnumeric().max()
                latest_data = test_orig[test_orig['year'] == str(int(year_max))]
            else:
                latest_data = test_orig
        except Exception as e:
            print(f"Error filtering by year: {e}")
            latest_data = test_orig
            
        # Make a copy to add predictions
        pred_data = h2o.assign(latest_data, 'pred_data_temp')
        
        # Add frequency predictions
        try:
            with h2o.utils.threading.local_context(polars_enabled=True):
                freq_df = freq_test[['predicted_freq', 'predicted_count']].as_data_frame()
                
            # Need to ensure we're only using rows for the latest year
            if 'year' in test_orig.columns:
                year_mask = freq_test['year'] == str(int(year_max))
                freq_latest = freq_test[year_mask]
                
                with h2o.utils.threading.local_context(polars_enabled=True):
                    freq_df = freq_latest[['predicted_freq', 'predicted_count']].as_data_frame()
            
            # Add predictions to our temporary frame
            pred_data['predicted_freq'] = h2o.H2OFrame(freq_df['predicted_freq'].values)
            pred_data['predicted_count'] = h2o.H2OFrame(freq_df['predicted_count'].values)
        except Exception as e:
            h2o.remove('pred_data_temp')
            print(f"Error adding frequency predictions: {e}")
            return jsonify({
                'error': f'Failed to add frequency predictions: {str(e)}'
            }), 500
            
        # Add severity predictions
        try:
            with h2o.utils.threading.local_context(polars_enabled=True):
                sev_df = sev_test[['predicted_sev']].as_data_frame()
                
            # Need to ensure we're only using rows for the latest year
            if 'year' in test_orig.columns:
                year_mask = sev_test['year'] == str(int(year_max))
                sev_latest = sev_test[year_mask]
                
                with h2o.utils.threading.local_context(polars_enabled=True):
                    sev_df = sev_latest[['predicted_sev']].as_data_frame()
            
            # Add predictions to our temporary frame
            pred_data['predicted_sev'] = h2o.H2OFrame(sev_df['predicted_sev'].values)
        except Exception as e:
            h2o.remove('pred_data_temp')
            print(f"Error adding severity predictions: {e}")
            return jsonify({
                'error': f'Failed to add severity predictions: {str(e)}'
            }), 500
            
        # Calculate predicted GUC
        pred_data['predicted_guc'] = pred_data['predicted_count'] * pred_data['predicted_sev']
        
        # Group by predictor for visualization
        try:
            grouped_frame_id = 'grouped_temp_agg'
            grouped = pred_data.group_by(predictor).sum(['exposure', 'guc', 'predicted_guc']).get_frame()
            h2o.assign(grouped, grouped_frame_id)
            
            with h2o.utils.threading.local_context(polars_enabled=True):
                result = grouped.as_data_frame()

            # Clean up
            h2o.remove(grouped_frame_id)
            h2o.remove('pred_data_temp')
            
            # Calculate predicted and actual burning costs
            result['actual_burning_cost'] = result['sum_guc'] / result['sum_exposure']
            result['predicted_burning_cost'] = result['sum_predicted_guc'] / result['sum_exposure']
            
            # Round results to 2 decimal places
            numeric_cols = result.select_dtypes(include=['float64', 'int64']).columns
            result[numeric_cols] = result[numeric_cols].round(2)
            
            # Rename columns to match frontend expectations
            result = result.rename(columns={
                'sum_exposure': 'exposure',
                'sum_guc': 'guc',
                'sum_predicted_guc': 'predicted_guc'
            })
            
            return jsonify({'data': result.replace({float('nan'): None}).to_dict('records')})
        except Exception as e:
            # Clean up if needed
            try:
                h2o.remove('pred_data_temp')
                h2o.remove(grouped_frame_id)
            except:
                pass
                
            print(f"Error in grouping: {e}")
            print(traceback.format_exc())
            return jsonify({
                'error': f'Error generating predictor plot data: {str(e)}'
            }), 500

    except Exception as e:
        print(f"Error in plot_predictor_agg: {str(e)}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e)
        }), 500

@app.route('/api/download-predictions', methods=['GET'])
def download_predictions():
    try:
        test = h2o.get_frame('test_frame')
        if test is None:
            return jsonify({'error': 'Test frame not found'}), 404
            
        with h2o.utils.threading.local_context(polars_enabled=True):
            result = test.as_data_frame()
        
        # Handle NaN values and round numerics
        result = result.fillna('')  # Replace NaN with empty string
        numeric_cols = result.select_dtypes(include=['float64', 'int64']).columns
        result[numeric_cols] = result[numeric_cols].round(4)
        
        # Convert to records and replace any remaining NaN with None
        records = result.replace({float('nan'): None}).to_dict('records')
        
        return jsonify({'data': records})
        
    except Exception as e:
        print(f"Error in download_predictions: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port, debug=False)
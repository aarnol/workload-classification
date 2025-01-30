% export_hb_filt_to_csv.m
% This script exports NIRS data from a Hb_filt nirs.core.Data object to individual CSV files for each participant.
% Each CSV file is named using the participant's ID (e.g., 'P07.csv').

% Ensure the Hb_filt object is loaded in the workspace before running this script.

% Determine the number of participants dynamically
numParticipants = size(Hb_filt, 1);

% Optional: Specify an output directory
% Uncomment and set your desired directory if needed
% outputDir = 'C:\NIRS_Exports\';
% Ensure the directory exists
% if ~exist(outputDir, 'dir')
%     mkdir(outputDir);
% end

% Loop through each participant
for p = 1:numParticipants
    try
        % Extract data matrix (samples x channels) for participant p
        data = Hb_filt(p,1).data; % Expected Size: 11521 x 178
        
        % Extract time vector for participant p
        time = Hb_filt(p,1).time; % Expected Size: 11521 x 1
        
        % Ensure time is a column vector
        if isrow(time)
            time = time';
        end
        
        % Validate dimensions of data and time
        if length(time) ~= size(data, 1)
            error('Mismatch between time vector length and data rows for participant %d.', p);
        end
        
        % Extract participant ID from demographics.values
        demographicsValues = Hb_filt(p,1).demographics.values;
        
        % Validate that demographicsValues is a cell containing a string
        if ~iscell(demographicsValues) || isempty(demographicsValues{1}) || ~ischar(demographicsValues{1}) && ~isstring(demographicsValues{1})
            error('Invalid demographics.values format for participant %d. Expected a cell containing a string ID.', p);
        end
        
        participantID = string(demographicsValues{1}); % Convert to string for consistency
        
        % Optional: Sanitize participantID to remove or replace invalid filename characters
        % Replace spaces with underscores and remove other invalid characters
        participantID = regexprep(participantID, '[<>:"/\\|?*]', '_');
        participantID = strrep(participantID, ' ', '_');
        
        % Combine time and data into one matrix
        combinedData = [time, data];
        
        % Create column names: 'Time', 'Ch1', 'Ch2', ..., 'Ch178'
        numChannels = size(data, 2);
        channelNames = strcat('Ch', string(1:numChannels));
        varNames = ['Time', channelNames];
        
        % Convert the combined data to a table with appropriate column headers
        T = array2table(combinedData, 'VariableNames', varNames);
        
        % Define the filename using the participant ID (e.g., 'P07.csv')
        filename = sprintf('data_csvs/%s.csv', participantID);
        
        % If using a specific output directory, uncomment the following line
        % filename = fullfile(outputDir, sprintf('%s.csv', participantID));
        
        % Write the table to a CSV file
        writetable(T, filename);
        
        % Display a confirmation message
        fprintf('Exported data for Participant %s to %s\n', participantID, filename);
        
    catch ME
        % Display an error message without stopping the entire script
        fprintf('Error exporting data for participant %d: %s\n', p, ME.message);
    end
end

% End of script

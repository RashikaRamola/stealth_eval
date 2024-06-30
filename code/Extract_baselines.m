%load('/Users/rashi/Documents/Academics/Research/CAFA4_eval/CAFA2/baselines/mfo/BN4S.mat')


%outputFileName = 'predicted_terms.tsv';
%pred2tsv(pred_path, outputFileName);


% ============
currentDir = '/data/common/CAFA4/baselines/'
outDir = '/data/common/CAFA4/processed_baselines/'
processDirectory(currentDir, outDir)

function processDirectory(currentDir, outDir)
    % List all files and folders in the current directoryx
    items = dir(currentDir);

    % Filter out '.' and '..' which are not useful
    items = items(~ismember({items.name}, {'.', '..', 'README.txt', 'bpo', 'cco', 'BB4S.mat'}));
    items
    for k = 1:length(items)
        item = items(k);
        itemPath = fullfile(currentDir, item.name);
        item.name
        if item.isdir
            % If it's a directory, recurse into it
            outputDir_nested = fullfile(outDir, item.name)
            mkdir(outputDir_nested)
            processDirectory(itemPath, outputDir_nested);
        else
            % If it's a file, process it
            itemPath
            file_name_split = split(item.name, ".")
            outputFileName = fullfile(outDir, file_name_split{1})
            pred2tsv(itemPath, outputFileName);
        end
    end
end

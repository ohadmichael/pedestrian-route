function output_table = run_python_script(python_command, output_file)
    cd '../Python/';

    system(['python ' python_command]);
    output_table = readtable(output_file, "ReadVariableNames", false);

    cd '../Matlab/';
end

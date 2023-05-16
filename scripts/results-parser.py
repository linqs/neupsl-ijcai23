import abc
import os

class AbstractResultsParser(abc.ABC):
    def __init__(self, results_dir, log_filename, additional_headers, specialized_headers=None):
        self.results_dir = results_dir
        self.log_filename = log_filename
        self.additional_headers = additional_headers
        self.specialized_headers = specialized_headers

    def parse_results(self):
        results = {}
        for experiment in sorted(os.listdir(self.results_dir)):
            if os.path.isfile(os.path.join(self.results_dir, experiment)):
                continue

            log_paths = self.get_log_paths(os.path.join(self.results_dir, experiment))
            results[experiment] = {'header': self.get_headers(log_paths[0], experiment) + self.additional_headers, 'rows': []}
            if self.specialized_headers is not None:
                results[experiment]['header'] += self.specialized_headers[experiment]
            for log_path in log_paths:
                parts = os.path.dirname(log_path.split(experiment + "/")[1]).split("/")
                results[experiment]['rows'].append([row.split("::")[1] for row in parts] + self.parse_log_path(log_path))

        self.print_results(results)

    def get_log_paths(self, path):
        log_paths = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if self.log_filename == file.split("/")[-1]:
                    log_paths.append(os.path.join(root,file))

        return sorted(log_paths)

    def get_headers(self, log_path, experiment):
        parts = os.path.dirname(log_path.split(experiment + "/")[1]).split("/")
        return [part.split("::")[0] for part in parts]

    @abc.abstractmethod
    def parse_log_path(self, log_path):
        pass

    def print_results(self, results):
        for experiment, result in sorted(results.items()):
            print("Experiment: %s" % (experiment,))
            print(' '.join(result['header']))
            for row in result['rows']:
                print(' '.join([str(value) for value in row]))
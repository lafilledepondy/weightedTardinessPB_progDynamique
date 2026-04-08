###################################################################################
# Functions required for the weighted tardiness single machine scheduling problem #
###################################################################################
class SchedulingInstance:
    """
    Represents a single-machine scheduling instance with weighted tardiness data.
    Attributes:
        nb_jobs (int): Number of jobs in the instance.
        processing_times (list[int]): Processing time for each job.
        weights (list[int]): Penalty weight for each job.
        due_dates (list[int]): Due date for each job.
        horizon (int): Total processing time of all jobs (planning horizon).
    Methods:
        from_file(file_path):
            Build and return a SchedulingInstance from a text file.
            Expected file format:
                - First non-empty line: number of jobs (integer).
                - Next non-empty lines: one job per line with
                  "processing_time due_date weight".
            Raises:
                ValueError: If the file is empty, malformed, or job count is inconsistent.
    """
    def __init__(self, nb_jobs, processing_times, weights, due_dates):
        self.nb_jobs=nb_jobs
        self.processing_times=processing_times
        self.weights=weights
        self.due_dates=due_dates
        self.horizon=sum(processing_times)
    
    # read from file
    @staticmethod
    def from_file(file_path):
        with open(file_path) as f:
            first_line = f.readline().strip()
            if first_line == "":
                raise ValueError("Input file is empty.")
            nb_jobs=int(first_line)
            # in each line, the first number is the processing time, the second number is the due date, the third number is the weight
            processing_times=[0]*nb_jobs
            weights=[0]*nb_jobs
            due_dates = [0]*nb_jobs
            job_idx = 0
            for line in f:
                stripped = line.strip()
                if stripped == "":
                    continue
                values = stripped.split()
                if len(values) < 3:
                    raise ValueError(f"Invalid job line: '{stripped}'. Expected 3 values.")
                if job_idx >= nb_jobs:
                    raise ValueError("File contains more jobs than declared in the first line.")
                processing_times[job_idx] = int(values[0])
                due_dates[job_idx] = int(values[1])
                weights[job_idx] = int(values[2])
                job_idx += 1

            if job_idx != nb_jobs:
                raise ValueError(
                    f"File contains {job_idx} jobs but first line declares {nb_jobs}."
                )

            return SchedulingInstance(nb_jobs, processing_times, weights, due_dates)




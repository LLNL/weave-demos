"""
This is the simplest DSV parser that's still useful.

The only thing you can have is a record id and a set of data;
"true" Sina Records are able to handle tags and units, files,
arbitrarily complex custom data (user_defined), and descriptive
relationship tuples.

The first column has to be the record id. Everything else is
arbitrary.

Written to use as little outside software as possible.
"""
import sys
import csv
import ast
import os
import json
import sina
from sina.datastore import create_datastore
from sina.model import Record, CurveSet
DELIMETER = "%"
csv.field_size_limit(sys.maxsize)


def record_from_csv(source_path):
    """Ingests CSV for simplest case."""
    with open(source_path) as source_csv:
        # Takes the CSV header and uses it to name our data.
        datareader = csv.reader(source_csv, delimiter=DELIMETER, quotechar='"')
        names = next(datareader)

        # Takes the rest and populates our record.
        # Note that we're keeping things simple here! Even though it's csv,
        # it's only a single record PER csv. We could do much more, but this
        # lets us sort of mock up a common user workflow (a file per run)
        # while still having an excuse to show a file "converter".
        record_data = next(datareader)
        record = Record(id=record_data[0], type="csv_rec")
        cycle_set = CurveSet("physics_cycle_series")
        encountered_series = False
        for index, entry in enumerate(record_data[1:]):
            nm = names[index+1]
            try:
                val = float(entry)
                record.data[names[index+1]] = {"value": val}
            except ValueError:
                # We know the UUID won't have [. Still a bit hacky.
                if "[" in entry:
                    val = ast.literal_eval(entry)
                    if not encountered_series:
                        cycle_set.add_independent("cycle", list(range(0, len(val))))
                        encountered_series = True
                    cycle_set.add_dependent(names[index+1], val)
                else:  # just a normal string
                    val = entry
                    record.data[names[index+1]] = {"value": val}
        record.add_curve_set(cycle_set)
        return record


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: scriptname source_path dest_sql")
    else:
        datastore = sina.connect(sys.argv[2])
        datastore.delete_all_contents(force="SKIP PROMPT")
        records = []
        for root, dirs, files in os.walk(sys.argv[1]):
            for file_name in files:
                if file_name.endswith("output.dsv"):
                    records.append(record_from_csv(os.path.join(root, file_name)))
        datastore.records.insert(records)

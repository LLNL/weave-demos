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
from sina.model import Record
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
        cs = record.add_curve_set("time_series")
        for index, entry in enumerate(record_data[1:]):
            nm = names[index+1]
            try:
                val = float(entry)
                record.data[nm] = {"value": val}
            except ValueError:
                # We know the UUID won't have [. Still a bit hacky.
                if "[" in entry:
                    val = ast.literal_eval(entry)
                    if nm == "time":
                        cs.add_independent(nm, val)
                    else:
                        cs.add_dependent(nm, val)
                else:  # just a normal string
                    val = entry
                    record.data[nm] = {"value": val}
        json_name = os.path.join(os.path.dirname(source_path), os.path.basename(source_path)[:-3]+"json")
        print(json_name)
        with open(json_name, "w") as f:
            f.write(record.to_json().decode())
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

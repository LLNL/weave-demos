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
import kosh
from sina.model import CurveSet
DELIMETER = "%"
csv.field_size_limit(sys.maxsize)


def dataset_from_csv(datastore, source_path):
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
        metadata = {}
        curve_set = "physics_cycle_series"
        cycle_set = CurveSet(curve_set)
        encountered_series = False
        for index, entry in enumerate(record_data[1:]):
            nm = names[index+1]
            try:
                val = float(entry)
                metadata[names[index+1]] = val
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
                    metadata[names[index+1]] = val

        ###########################
        # Original non-threadsafe #
        ###########################

        # dataset = datastore.create(id = record_data[0],
        #                            metadata = metadata)
        # dataset.add_curve(cycle_set.__dict__['raw']['independent']['cycle']['value'], curve_set, 'cycle', independent=True)
        # dataset.add_curve(cycle_set.__dict__['raw']['dependent']['time']['value'], curve_set, 'time', independent=False)
        # dataset.add_curve(cycle_set.__dict__['raw']['dependent']['x_pos']['value'], curve_set, 'x_pos', independent=False)
        # dataset.add_curve(cycle_set.__dict__['raw']['dependent']['y_pos']['value'], curve_set, 'y_pos', independent=False)
        # dataset.add_curve(cycle_set.__dict__['raw']['dependent']['z_pos']['value'], curve_set, 'z_pos', independent=False)

        ##############
        # Threadsafe #
        ##############

        # The default safe_create uses the decorator settings `@threadsafe_decorators.threadsafe_call(1, 0)`
        # This means retry to connect to the store 1 time and wait 0 seconds in between each retry
        # Sometimes these settings are not enough and a custom threadsafe method needs to be created
        # A custom example can be seen below with `@threadsafe_decorators()`
        # dataset =  kosh.utils.threadsafe.safe_create(datastore,
        #                                              id = record_data[0],
        #                                              metadata = metadata)

        # See Kosh examples: `Example_ThreadSafe.ipynb`
        # Retry to connect to the store 10000 times and wait 2 seconds in between each retry
        # We have this large retry number since these large amount of simulations finish almost instantaneously
        # and since they are all retrying to connect at the same time that can also cause issues
        @kosh.utils.threadsafe_decorators.threadsafe_call(10000, 2)
        def my_custom_create(store, **kw_args):
            return store.create(**kw_args)

        dataset = my_custom_create(datastore,
                                   id=record_data[0],
                                   metadata=metadata)

        # We also need to create a custom threadsafe method that adds data to a dataset
        @kosh.utils.threadsafe_decorators.threadsafe_call(10000, 2)
        def my_custom_add_curve(dataset, **kw_args):
            return dataset.add_curve(**kw_args)

        my_custom_add_curve(dataset,
                            curve=cycle_set.__dict__['raw']['independent']['cycle']['value'],
                            curve_set=curve_set,
                            curve_name='cycle',
                            independent=True)
        my_custom_add_curve(dataset,
                            curve=cycle_set.__dict__['raw']['dependent']['time']['value'],
                            curve_set=curve_set,
                            curve_name='time',
                            independent=False)
        my_custom_add_curve(dataset,
                            curve=cycle_set.__dict__['raw']['dependent']['x_pos']['value'],
                            curve_set=curve_set,
                            curve_name='x_pos',
                            independent=False)
        my_custom_add_curve(dataset,
                            curve=cycle_set.__dict__['raw']['dependent']['y_pos']['value'],
                            curve_set=curve_set,
                            curve_name='y_pos',
                            independent=False)
        my_custom_add_curve(dataset,
                            curve=cycle_set.__dict__['raw']['dependent']['z_pos']['value'],
                            curve_set=curve_set,
                            curve_name='z_pos',
                            independent=False)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: scriptname source_path dest_sql")
    else:

        datastore = kosh.connect(sys.argv[2])

        records = []
        for root, dirs, files in os.walk(sys.argv[1]):
            for file_name in files:
                if file_name.endswith("output.dsv"):
                    dataset_from_csv(datastore, os.path.join(root, file_name))

from sdv.io.local import CSVHandler
from sdv.metadata import Metadata
from sdv.single_table import GaussianCopulaSynthesizer



connector = CSVHandler()
data = connector.read(
    folder_name='datasets/',
    file_names=['call-data-06-2026.csv'],
    keep_leading_zeros=True
)



metadata = Metadata.detect_from_dataframes(data)


# metadata.visualize()
# metadata.save_to_json('my_final_metadata.json')


synthesizer = GaussianCopulaSynthesizer(metadata)
synthesizer.fit(data['call-data-06-2026'])
synthetic_data = synthesizer.sample(num_rows=10)
print(synthetic_data)
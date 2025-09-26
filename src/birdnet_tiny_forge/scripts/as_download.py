from pathlib import Path

import click
import joblib
import pandas as pd
from audioset_download import Downloader


_OUTDIR = Path(__file__).parents[3] / "data" / "01_raw" / "audio_clips"


class DownloaderWithLimit(Downloader):

  def download(
        self,
        format: str = 'vorbis',
        quality: int = 5,
        limit: int = 100,
    ):
        """
        NOTE: this is exactly like the original method, but we limit the clips.

        This method downloads the dataset using the provided parameters.
        :param format: format of the audio file (vorbis, mp3, m4a, wav), default is vorbis
        :param quality: quality of the audio file (0: best, 10: worst), default is 5
        """

        self.format = format
        self.quality = quality

        # Load the metadata
        metadata = pd.read_csv(
            f"http://storage.googleapis.com/us_audioset/youtube_corpus/v1/csv/{self.download_type}_segments.csv",
            sep=', ',
            skiprows=3,
            header=None,
            names=['YTID', 'start_seconds', 'end_seconds', 'positive_labels'],
            engine='python'
        )
        if self.labels is not None:
            self.real_labels = [self.display_to_machine_mapping[label] for label in self.labels]
            metadata = metadata[metadata['positive_labels'].apply(lambda x: any([label in x for label in self.real_labels]))]
        # remove " in the labels
        metadata['positive_labels'] = metadata['positive_labels'].apply(lambda x: x.replace('"', ''))
        metadata['positive_labels_list'] = metadata['positive_labels'].str.split(',')
        metadata_exploded = metadata.explode('positive_labels_list')
        metadata_exploded['positive_labels_list'] = metadata_exploded['positive_labels_list'].str.strip()

        # Sample n records for each individual label
        sampled_exploded = (
            metadata_exploded
            .query("positive_labels_list in @self.real_labels")
            .groupby("positive_labels_list")
            .sample(n=limit, replace=False)
        )

        # Get unique original rows (since same row might be selected multiple times for different labels)
        metadata = metadata.loc[sampled_exploded.index.unique()]

        # Keep one label per example, so we don't create spurious folders with the other labels
        metadata["positive_labels"] = sampled_exploded.loc[sampled_exploded.index.unique(), 'positive_labels_list']
        metadata = metadata.drop("positive_labels_list", axis=1)
        metadata = metadata.reset_index(drop=True)

        print(f'Downloading {len(metadata)} files...')
        # Download the dataset
        joblib.Parallel(n_jobs=self.n_jobs, verbose=10)(
            joblib.delayed(self.download_file)(metadata.loc[i, 'YTID'], metadata.loc[i, 'start_seconds'], metadata.loc[i, 'end_seconds'], metadata.loc[i, 'positive_labels']) for i in range(len(metadata))
        )
        print('Done.')


@click.command()
@click.option('--outdir', default=_OUTDIR, type=click.Path(exists=True, file_okay=False, dir_okay=True, path_type=Path), help='Root directory for downloads')
@click.option('--labels-file', required=True, type=click.Path(exists=True, dir_okay=False, path_type=Path), help='File containing labels list')
@click.option('--format', default='mp3', type=click.Choice(['mp3', 'wav', 'flac', 'm4a']), help='Audio format')
@click.option('--quality', default=5, type=click.IntRange(0, 10), help='Audio quality (0-10)')
@click.option('--limit', default=100, type=click.IntRange(1), help='Maximum number of downloads')
@click.option('--n-jobs', default=2, type=click.IntRange(1), help='Number of parallel jobs')
@click.option('--download-type', default='unbalanced_train', type=click.Choice(['unbalanced_train', 'balanced_train', 'eval']), help='Dataset type to download')
@click.option('--copy-and-replicate/--no-copy-and-replicate', default=False, help='Enable copy and replicate mode')
def as_download(
        outdir,
        labels_file,
        format="mp3",
        quality=5,
        limit: int = 100,
        n_jobs=2,
        download_type='unbalanced_train',
        copy_and_replicate=False,

):
    with Path(labels_file).open("r") as f:
        labels = [x.strip() for x in f.readlines()]
    d = DownloaderWithLimit(
        root_path=outdir,
        labels=labels,
        n_jobs=n_jobs,
        download_type=download_type,
        copy_and_replicate=copy_and_replicate
    )
    d.download(format=format, quality=quality, limit=limit)


if __name__ == "__main__":
    as_download()
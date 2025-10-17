import gzip
import json
import os
from typing import Dict, List, Optional, Union

import attr
from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.core.registry import registry
from habitat.core.utils import not_none_validator
from habitat.datasets.pointnav.pointnav_dataset import ALL_SCENES_MASK
from habitat.datasets.utils import VocabDict
from habitat.tasks.nav.nav import NavigationGoal
from habitat.tasks.vln.vln import InstructionData, VLNEpisode
import random
from tqdm import tqdm
#random.seed(0)
import time

DEFAULT_SCENE_PATH_PREFIX = "data/scene_datasets/"
ALL_LANGUAGES_MASK = "*"
ALL_ROLES_MASK = "*"


@attr.s(auto_attribs=True)
class ExtendedInstructionData:
    instruction_text: str = attr.ib(default=None, validator=not_none_validator)
    instruction_id: Optional[str] = attr.ib(default=None)
    language: Optional[str] = attr.ib(default=None)
    annotator_id: Optional[str] = attr.ib(default=None)
    edit_distance: Optional[float] = attr.ib(default=None)
    timed_instruction: Optional[List[Dict[str, Union[float, str]]]] = attr.ib(
        default=None
    )
    instruction_tokens: Optional[List[str]] = attr.ib(default=None)
    split: Optional[str] = attr.ib(default=None)


@attr.s(auto_attribs=True, kw_only=True)
class VLNExtendedEpisode(VLNEpisode):
    goals: Optional[List[NavigationGoal]] = attr.ib(default=None)
    reference_path: Optional[List[List[float]]] = attr.ib(default=None)
    instruction: ExtendedInstructionData = attr.ib(
        default=None, validator=not_none_validator
    )
    trajectory_id: Optional[Union[int, str]] = attr.ib(default=None)


@registry.register_dataset(name="VLN-CE-v1")
class VLNCEDatasetV1(Dataset):
    r"""Class inherited from Dataset that loads a Vision and Language
    Navigation dataset.
    """

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict

    @staticmethod
    def check_config_paths_exist(config: Config) -> bool:
        return os.path.exists(
            config.DATA_PATH.format(split=config.SPLIT)
        ) and os.path.exists(config.SCENES_DIR)

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []

        if config is None:
            return

        dataset_filename = config.DATA_PATH.format(split=config.SPLIT)
        with gzip.open(dataset_filename, "rt") as f:
            self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

        if config.EPISODES_ALLOWED is not None:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set([ int(id) for id in config.EPISODES_ALLOWED])
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)
        self.instruction_vocab = VocabDict(
            word_list=deserialized["instruction_vocab"]["word_list"]
        )

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.instruction = InstructionData(**episode.instruction)
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)
            self.episodes.append(episode)

        random.shuffle(self.episodes)


@registry.register_dataset(name="RxR-VLN-CE-v1")
class RxRVLNCEDatasetV1(Dataset):
    r"""Loads the RxR VLN-CE Dataset."""

    episodes: List[VLNEpisode]
    instruction_vocab: VocabDict
    annotation_roles: List[str] = ["guide", "follower"]
    languages: List[str] = ["en-US", "en-IN", "hi-IN", "te-IN"]

    @staticmethod
    def _scene_from_episode(episode: VLNEpisode) -> str:
        r"""Helper method to get the scene name from an episode.  Assumes
        the scene_id is formated /path/to/<scene_name>.<ext>
        """
        return os.path.splitext(os.path.basename(episode.scene_id))[0]

    @staticmethod
    def _language_from_episode(episode: VLNExtendedEpisode) -> str:
        return episode.instruction.language

    @classmethod
    def get_scenes_to_load(cls, config: Config) -> List[str]:
        r"""Return a sorted list of scenes"""
        assert cls.check_config_paths_exist(config)
        dataset = cls(config)
        return sorted(
            {cls._scene_from_episode(episode) for episode in dataset.episodes}
        )

    @classmethod
    def extract_roles_from_config(cls, config: Config) -> List[str]:
        if ALL_ROLES_MASK in config.ROLES:
            return cls.annotation_roles
        assert set(config.ROLES).issubset(set(cls.annotation_roles))
        return config.ROLES

    @classmethod
    def check_config_paths_exist(cls, config: Config) -> bool:
        return all(
            os.path.exists(
                config.DATA_PATH.format(split=config.SPLIT, role=role)
            )
            for role in cls.extract_roles_from_config(config)
        ) and os.path.exists(config.SCENES_DIR)

    def __init__(self, config: Optional[Config] = None) -> None:
        self.episodes = []
        self.config = config

        if config is None:
            return

        for role in self.extract_roles_from_config(config):
            with gzip.open(
                config.DATA_PATH.format(split=config.SPLIT, role=role), "rt"
            ) as f:
                self.from_json(f.read(), scenes_dir=config.SCENES_DIR)

        if ALL_SCENES_MASK not in config.CONTENT_SCENES:
            scenes_to_load = set(config.CONTENT_SCENES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._scene_from_episode(episode) in scenes_to_load
            ]

        if ALL_LANGUAGES_MASK not in config.LANGUAGES:
            languages_to_load = set(config.LANGUAGES)
            self.episodes = [
                episode
                for episode in self.episodes
                if self._language_from_episode(episode) in languages_to_load
            ]

        if config.EPISODES_ALLOWED is not None:
            ep_ids_before = {ep.episode_id for ep in self.episodes}
            ep_ids_to_purge = ep_ids_before - set(config.EPISODES_ALLOWED)
            self.episodes = [
                episode
                for episode in self.episodes
                if episode.episode_id not in ep_ids_to_purge
            ]

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:

        deserialized = json.loads(json_str)

        for episode in deserialized["episodes"]:
            episode = VLNExtendedEpisode(**episode)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)
            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )
            episode.instruction.split = self.config.SPLIT
            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)

            self.episodes.append(episode)

@registry.register_dataset(name="VLN-CE-v1-3DFF")
class VLNCEDatasetV1_3DFF(Dataset):
    r"""Loads the HM3D Dataset for 3DFF pretraining."""

    episodes: List[VLNEpisode]
       
    def get_scenes_to_load(self):
        r"""Return a sorted list of scenes"""
        return self.episodes

    def __init__(self, config: Optional[Config] = None) -> None:

        self.config = config
        self.episodes = []
        seed = int(time.time())
        random.seed(seed)

        if seed % 2 == 0:
            hm3d_dir = "data/scene_datasets/hm3d"
            scene_ids = os.listdir(os.path.join(hm3d_dir, 'train')) #+ os.listdir(os.path.join(hm3d_dir, 'val')) # Remove the val split
            random.shuffle(scene_ids)

            for i in range(len(scene_ids)):
                scene_id = scene_ids[i]
                if int(scene_id.split('-')[0]) < 800:
                    split = 'train'
                else:
                    split = 'val'
                scene_id = os.path.join(hm3d_dir, split, scene_id, scene_id.split('-')[-1]+'.basis.glb')
                scene_ids[i] = scene_id

            count = 0
            print("Loading the HM3D dataset...")
            annotated_scenes = ['L5QEsaVqwrY', 'mL8ThkuaVTM', 'ACZZiU6BXLz', 'R9fYpvCUkV7', 'qk9eeNeR4vw', 'Jfyvj3xn2aJ', 'nACV8wLu1u5', 'gjhYih4upQ9', 'GsQBY83r3hb', 'g8Xrdbe9fir', '6HRFAUDqpTb', 'xgLmjqzoAzF', '6imZUJGRUq4', 'GLAQ4DNUx5U', 'u5atqC7vRCY', 'b3WpMbPFB6q', 'bB6nKqfsb1z', 'HeSYRw7eMtG', 'h1zeeAwLh9Z', 'BAbdmeyTvMZ', 'CthA7sQNTPK', '6YtDG3FhNvx', 'TEEsavR23oF', 'nS8T59Aw3sf', 'oEPjPNSPmzL', 'LT9Jq6dN3Ea', 'svBbv1Pavdk', 'ziup5kvtCCR', 'yX5efd48dLf', 'zt1RVoi7PcG', '92vYG1q49FY', '3XYAD64HpDr', 'iigzG1rtanx', 'Dd4bFSTQ8gi', 'DBBESbk4Y3k', 'UuwwmrTsfBN', 'fK2vEV32Lag', 'g7hUFVNac26', '6s7QHgap2fW', 'PE6kVEtrxtj', 'QVAA6zecMHu', 'gQ3xxshDiCz', 'XfUxBGTFQQb', 'q3zU7Yy5E5s', '5biL7VEkByM', 'hWDDQnSDMXb', 'LVgQNuK8vtv', 'RTV2n6fXB2w', 'GTV2Y73Sn5t', 'ceJTwFNjqCt', 'qZ4B7U6XE5Y', 'HfMobPm86Xn', 'zepmXAdrpjR', 'oPj9qMxrDEa', 'YmWinf3mhb5', 'bHKTDQFJxTw', '741Fdj7NLF9', 'sX9xad6ULKc', 'XVSZJAtHKdi', 'TSJmdttd2GV', 'TYDavTf8oyy', 'W16Bm4ysK8v', 'wcojb4TFT35', 'LcAd9dhvVwh', 'X6Pct1msZv5', '2Pc8W48bu21', 'wPLokgvCnuk', 'H8rQCnvBgo6', 'bdp1XNEdvmW', 'GGBvSFddQgs', 'j6fHrce9pHR', 'kA2nG18hCAr', 'kJxT5qssH4H', 'XiJhRLvpKpX', 'NEVASPhcrxR', 'nGhNxKrgBPb', 'j2EJhFEQGCL', 'URjpCob8MGw', '5cdEh9F2hJL', 'PPTLa8SkUfo', 'GPyDUnjwZQy', '4ok3usBNeis', 'NGyoyh91xXJ', 'S7uMvxjBVZq', 'MVVzj944atG', 'MHPLjHsuG27', 'oahi4u45xMf', 'q5QZSEeHe5g', 'y9hTuugGdiq', 'k1cupFYWXJ6', 'h6nwVLpAKQz', 'erXNfWVjqZ8', 'xWvSkKiWQpC', 'fRZhp6vWGw7', 'aRKASs4e8j1', 'HZ2iMMBsBQ9', 'vLpv2VX547B', 'W9YAR9qcuvN', 'v7DzfFFEpsD', 'mv2HUxq3B53', 'RaYrxWt5pR1', 'qz3829g1Lzf', 'gQgtJ9Stk5s', 'gmuS7Wgsbrx', 'ooq3SnvC79d', 'qgZhhx1MpTi', 'CrMo8WxCyVb', 'oStKKWkQ1id', 'xAHnY3QzFUN', 'KjZrPggnHm8', '1S7LAXRdDqK', 'JNiWU5TZLtt', '226REUyJh2K', 'mt9H8KcxRKD', 'a8BtkwhxdRV', 'iKFn6fzyRqs', 'VSxVP19Cdyw', 'CQWES1bawee', 'Z2DQddYp1fn', 'XB4GS9ShBRE', 'Nfvxx8J5NCo', 'GtM3JtRvvvR', 'dQrLTxHvLXU', 'ixTj1aTMup2', 'JptJPosx1Z6', '4vwGX7U38Ux', 'DsEJeNPcZtE', '9h5JJxM6E5S', 'QN2dRqwd84J', 'E1NrAhMoqvB', '8B43pG641ff', '5Kw4nGdqYtS', 'NtnvZSMK3en', '77mMEyxhs44', 'bxsVRursffK', 'VoVGtfYrpuQ', 'YHmAkqgwe2p', 'iLDo95ZbDJq', 'DqJKU7YU7dA', 'NPHxDe6VeCc', 'XYyR54sxe6b', 'eF36g7L6Z9M', 'ZNanfzgCdm3', 'WhNyDTnd9g5', 'SgkmkWjjmDJ', 'cvZr5TUy5C5', 'YJDUB7hWg9h', 'ENiCjXWB6aQ', 'Wo6kuutE9i7', 'iePHCSf119p', 'DoSbsoo4EAg', '7MXmsvcQjpJ', 'u9rPN5cHWBg', 'YMNvYDhK8mB', 'FnDDfrBZPhh', 'U3oQjwTuMX8', 'vDfkYo5VqEQ', 'VBzV5z6i1WS', 'zUG6FL9TYeR', 'HxmXPBbFCkH', 'qyAac8rV8Zk', 'fxbzYAGkrtm', 'YY8rqV6L6rf', 'FRQ75PjD278', 'bCPU9suPUw9', 'QaLdnwvtxbs', 'wsAYBFtQaL7', '1UnKg1rAb8A', 'w8GiikYuFRk', '1xGrZPxG1Hz', 'HY1NcmCgn3n', 'yHLr6bvWsVm', '3CBBjsNkhqW', 'HkseAnWCgqk', 'p53SfW6mjZe', '8wJuSPJ9FXG', 'DYehNKdT76V', 'DNWbUAJYsPy', 'yr17PDCnDDW']

            for scene_id in tqdm(scene_ids):
                if scene_id.split("/")[-1].split(".")[0] not in annotated_scenes:
                    continue
                if "train" in scene_id:
                    with gzip.open(
                            "data/datasets/pointnav/hm3d/v1/train/content/"+scene_id.split("/")[-1].split(".")[0]+'.json.gz', "rt"
                        ) as f:
                            self.from_json(f.read())
                #else:
                #    with gzip.open(
                #            "data/datasets/pointnav/hm3d/v1/val/content/"+scene_id.split("/")[-1].split(".")[0]+'.json.gz', "rt"
                #        ) as f:
                #            self.from_json(f.read())
                count += 1
                if count == 16:
                    break
        
        else:
            mp3d_dir = "data/datasets/pointnav/mp3d/train/content"
            scene_ids = os.listdir(mp3d_dir)
            random.shuffle(scene_ids)

            for i in range(len(scene_ids)):
                scene_id = scene_ids[i].split(".")[0]
                scene_id = os.path.join(mp3d_dir, scene_id, scene_id+'.basis.glb')
                scene_ids[i] = scene_id

            count = 0
            print("Loading the MP3D dataset...")
            
            for scene_id in tqdm(scene_ids):
                with gzip.open(
                        "data/datasets/pointnav/mp3d/train/content/"+scene_id.split("/")[-1].split(".")[0]+'.json.gz', "rt"
                    ) as f:
                        self.from_json(f.read())
                #else:
                #    with gzip.open(
                #            "data/datasets/pointnav/mp3d/val/content/"+scene_id.split("/")[-1].split(".")[0]+'.json.gz', "rt"
                #        ) as f:
                #            self.from_json(f.read())
                count += 1
                if count == 16:
                    break
        
        return None


    def from_json(
        self, json_str: str
    ) -> None:

        deserialized = json.loads(json_str)
        random.shuffle(deserialized["episodes"])
        for episode in deserialized["episodes"]:
            if 'data/scene_datasets/' not in episode['scene_id']:
                episode['scene_id'] = 'data/scene_datasets/'+episode['scene_id']
            # The instruction is not needed, just for running the code
            episode['instruction'] = {'instruction_id': '0', 'instruction_text': "", 'language': 'en-US', 'annotator_id': '0', 'edit_distance': 0., 'instruction_tokens': 6}
            
            episode = VLNExtendedEpisode(**episode)
            episode.instruction = ExtendedInstructionData(
                **episode.instruction
            )

            if episode.goals is not None:
                for g_index, goal in enumerate(episode.goals):
                    episode.goals[g_index] = NavigationGoal(**goal)

            self.episodes.append(episode)
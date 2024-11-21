import unittest
from llm_client import MessageHistory


class TestMessageHistoryTransactions(unittest.TestCase):

    def setUp(self):
        self.history = MessageHistory()

    def test_start_transaction(self):
        self.history.add("system", "System message")
        self.history.start_transaction()
        self.assertEqual(len(self.history.history_stack), 1)
        self.assertEqual(self.history.history_stack[0][0].content, "System message")

    def test_rollback_transaction(self):
        self.history.add("system", "System message")
        self.history.start_transaction()
        self.history.add("user", "User message")
        self.history.rollback()
        self.assertEqual(len(self.history.history), 1)
        self.assertEqual(self.history.history[0].content, "System message")

    def test_commit_transaction(self):
        self.history.add("system", "System message")
        self.history.start_transaction()
        self.history.add("user", "User message")
        self.history.commit()
        self.assertEqual(len(self.history.history_stack), 0)
        self.assertEqual(len(self.history.history), 2)

    def test_rollback_without_transaction(self):
        with self.assertRaises(ValueError):
            self.history.rollback()

    def test_commit_without_transaction(self):
        with self.assertRaises(ValueError):
            self.history.commit()

    def test_multiple_transactions(self):
        self.history.add("system", "System message")
        self.history.start_transaction()
        self.history.add("user", "User message 1")
        self.history.start_transaction()
        self.history.add("user", "User message 2")
        self.history.rollback()
        self.assertEqual(len(self.history.history), 2)
        self.assertEqual(self.history.history[1].content, "User message 1")
        self.history.rollback()
        self.assertEqual(len(self.history.history), 1)
        self.assertEqual(self.history.history[0].content, "System message")


class TestMessageHistoryApiFormat(unittest.TestCase):

    def setUp(self):
        self.history = MessageHistory()

    def test_to_api_format_basic(self):
        self.history.add("system", "System message")
        self.history.add("user", "User message")
        self.history.add("assistant", "Assistant message")
        self.history.add("user", "User message 2")

        expected_output = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "user", "content": "User message 2"}
        ]

        self.assertEqual(self.history.to_api_format(), expected_output)

    def test_to_api_format_with_agent(self):
        self.history.add("system", "System message")
        self.history.add("agent1", "User message")
        self.history.add("agent2", "Assistant message")
        self.history.add("agent1", "User message 2")

        expected_output = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "user", "content": "User message 2"}
        ]

        self.assertEqual(self.history.to_api_format(agent="agent2"), expected_output)

    def test_to_api_format_with_agent_complex(self):
        self.history.add("system", "System message")
        self.history.add("user", "Prompt")
        self.history.add("agent1", "User message")
        self.history.add("agent2", "Assistant message")
        self.history.add("agent1", "User message 2")
        self.history.add("agent2", "Assistant message 2")
        self.history.add("agent1", "User message 3")

        expected_output = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "Prompt\nUser message"},
            {"role": "assistant", "content": "Assistant message"},
            {"role": "user", "content": "User message 2"},
            {"role": "assistant", "content": "Assistant message 2"},
            {"role": "user", "content": "User message 3"}
        ]

        self.assertEqual(self.history.to_api_format(agent="agent2"), expected_output)

    def test_multiple_system_messages_with_agents(self):
        self.history.add("system_agent1", "System message")
        self.history.add("system_agent2", "Another system message")
        self.history.add("user", "User message")

        expected_output = [
            {"role": "system", "content": "System message"},
            {"role": "user", "content": "User message"}
        ]

        self.assertEqual(self.history.to_api_format(agent="agent1"), expected_output)

    def test_to_api_format_empty_history(self):
        with self.assertRaises(ValueError):
            self.history.to_api_format()

    def test_to_api_format_no_system_message(self):
        self.history.add("user", "User message")
        self.history.add("assistant", "Assistant message")

        with self.assertRaises(ValueError):
            self.history.to_api_format()

    def test_to_api_format_last_message_not_user(self):
        self.history.add("system", "System message")
        self.history.add("user", "User message")
        self.history.add("assistant", "Assistant message")
        self.history.add("assistant", "Another assistant message")

        with self.assertRaises(ValueError):
            self.history.to_api_format()


if __name__ == '__main__':
    unittest.main()

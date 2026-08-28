from agent import NoMemoryAgent, RuleBasedClient


def test_no_memory_abstains_without_context():
    agent = NoMemoryAgent(RuleBasedClient())
    assert agent.answer("我的项目名称是什么？") == "UNKNOWN"
